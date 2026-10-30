import * as cdk from 'aws-cdk-lib';
import { Template } from 'aws-cdk-lib/assertions';
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

  // Every worker, the fetch loop included, is an ECS service: the only Lambda is CDK's ASG drain hook, and nothing is
  // scheduled outside the fetch loop.
  template.resourceCountIs('AWS::Lambda::Function', 1);
  template.resourceCountIs('AWS::Events::Rule', 0);
  template.resourceCountIs('AWS::S3::Bucket', 0);
});
