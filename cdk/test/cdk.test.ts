import * as cdk from 'aws-cdk-lib';
import { Template } from 'aws-cdk-lib/assertions';
import { Stage } from '@gnome-trading-group/gnome-shared-cdk';
import { ClassifierStack } from '../lib/stacks/classifier-stack';

test('ClassifierStack has Step Functions state machine, three Lambdas, and EventBridge rule', () => {
  const app = new cdk.App();
  const stack = new ClassifierStack(app, 'TestStack', { stage: Stage.PROD, slackChannel: '' });
  const template = Template.fromStack(stack);

  template.resourceCountIs('AWS::Lambda::Function', 3);
  template.resourceCountIs('AWS::StepFunctions::StateMachine', 1);
  template.resourceCountIs('AWS::Events::Rule', 1);
});
