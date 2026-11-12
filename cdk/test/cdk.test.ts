import * as cdk from 'aws-cdk-lib';
import { Match, Template } from 'aws-cdk-lib/assertions';
import { Stage } from '@gnome-trading-group/gnome-shared-cdk';
import { ClassifierStack } from '../lib/stacks/classifier-stack';

// Use an account/region that has cached VPC context in cdk.context.json
const env = { account: '443370708724', region: 'us-east-1' };

test('ClassifierStack creates SQS queues, SNS topic, ECS cluster, and ElastiCache', () => {
  const app = new cdk.App();
  const stack = new ClassifierStack(app, 'TestStack', { stage: Stage.PROD, slackChannel: 'test-channel', env });
  const template = Template.fromStack(stack);

  // 3 main queues (contracts, entities, slack) + 3 DLQs
  template.resourceCountIs('AWS::SQS::Queue', 6);

  // 1 notifications topic + 1 ASG ECS drain hook topic (created by CDK)
  template.resourceCountIs('AWS::SNS::Topic', 2);
  // 1 slack-queue subscription + 1 ASG drain hook subscription (created by CDK)
  template.resourceCountIs('AWS::SNS::Subscription', 2);

  // One ECS service per queue-polling worker (normalize, relationships, notify) plus the fetch loop
  template.resourceCountIs('AWS::ECS::Cluster', 1);
  template.resourceCountIs('AWS::ECS::TaskDefinition', 4);
  template.resourceCountIs('AWS::ECS::Service', 4);

  // Redis cache
  template.resourceCountIs('AWS::ElastiCache::CacheCluster', 1);

  // Every worker, the fetch loop included, is an ECS service: the only Lambdas are CDK's ASG drain hook and the
  // custom-resource handler, and the only rule routes task crashes to their log group; nothing is scheduled outside
  // the fetch loop.
  template.resourceCountIs('AWS::Lambda::Function', 2);
  template.resourceCountIs('AWS::Events::Rule', 1);
  template.resourceCountIs('AWS::S3::Bucket', 0);
});

test('Fetch task limit is 3072 MiB (2048 MiB was OOM-killed)', () => {
  const app = new cdk.App();
  const stack = new ClassifierStack(app, 'TestStack', { stage: Stage.PROD, slackChannel: 'test-channel', env });
  const template = Template.fromStack(stack);

  const fetchTasks = Object.values(template.findResources('AWS::ECS::TaskDefinition')).filter((r: any) =>
    r.Properties.ContainerDefinitions.some((c: any) => c.Name === 'FetchContainer'));
  expect(fetchTasks).toHaveLength(1);
  expect((fetchTasks[0] as any).Properties.ContainerDefinitions[0].Memory).toBe(3072);
});

test('Worker crashes are counted per service from ECS task stops the container caused', () => {
  const app = new cdk.App();
  const stack = new ClassifierStack(app, 'TestStack', { stage: Stage.PROD, slackChannel: 'test-channel', env });
  const template = Template.fromStack(stack);

  template.hasResourceProperties('AWS::Events::Rule', {
    EventPattern: Match.objectLike({
      source: ['aws.ecs'],
      'detail-type': ['ECS Task State Change'],
      detail: Match.objectLike({ lastStatus: ['STOPPED'], stopCode: ['EssentialContainerExited'] }),
    }),
  });
  const crashFilters = Object.values(template.findResources('AWS::Logs::MetricFilter')).filter((r: any) =>
    r.Properties.MetricTransformations[0].MetricName.endsWith('WorkerCrashes'));
  expect(crashFilters.map((r: any) => r.Properties.MetricTransformations[0].MetricName).sort())
    .toEqual(['FetchWorkerCrashes', 'NormalizeWorkerCrashes', 'NotifyWorkerCrashes', 'RelationshipsWorkerCrashes']);
  expect(stack.workerCrashes.Relationships).toBeDefined();
});

test('The worker ASG has AZ rebalancing suspended', () => {
  const app = new cdk.App();
  const stack = new ClassifierStack(app, 'TestStack', { stage: Stage.PROD, slackChannel: 'test-channel', env });
  const template = Template.fromStack(stack);

  const suspends = Object.values(template.findResources('Custom::AWS')).filter((r: any) =>
    JSON.stringify(r.Properties.Update ?? r.Properties.Create).includes('AZRebalance'));
  expect(suspends).toHaveLength(1);
  expect(JSON.stringify((suspends[0] as any).Properties.Update)).toContain('suspendProcesses');
});
