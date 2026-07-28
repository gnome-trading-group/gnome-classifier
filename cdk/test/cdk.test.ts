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

  // 4 main queues + 4 DLQs
  template.resourceCountIs('AWS::SQS::Queue', 8);

  // 1 notifications topic + 1 ASG ECS drain hook topic (created by CDK)
  template.resourceCountIs('AWS::SNS::Topic', 2);
  // 1 slack-queue subscription + 1 ASG drain hook subscription (created by CDK)
  template.resourceCountIs('AWS::SNS::Subscription', 2);

  // One ECS service per queue-polling worker (normalize, embed, relationships, notify)
  template.resourceCountIs('AWS::ECS::Cluster', 1);
  template.resourceCountIs('AWS::ECS::TaskDefinition', 4);
  template.resourceCountIs('AWS::ECS::Service', 4);

  // Redis cache
  template.resourceCountIs('AWS::ElastiCache::CacheCluster', 1);

  // S3 cache bucket
  template.resourceCountIs('AWS::S3::Bucket', 1);

  // 1 ASG drain hook Lambda (CDK) + 2 application Lambdas (fetch, resolve)
  template.resourceCountIs('AWS::Lambda::Function', 3);
  template.resourceCountIs('AWS::StepFunctions::StateMachine', 0);
  // 2 EventBridge schedules: fetch (every 5 min) + resolve (every 15 min)
  template.resourceCountIs('AWS::Events::Rule', 2);
});
