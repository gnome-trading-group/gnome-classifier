import * as cdk from 'aws-cdk-lib';
import * as sfn from 'aws-cdk-lib/aws-stepfunctions';
import * as sns from 'aws-cdk-lib/aws-sns';
import { Construct } from 'constructs';
import { MonitoringFacade, SnsAlarmActionStrategy } from 'cdk-monitoring-constructs';

interface Props extends cdk.StackProps {
  stateMachine: sfn.IStateMachine;
}

export class MonitoringStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props: Props) {
    super(scope, id, props);

    const slackSnsTopic = sns.Topic.fromTopicArn(
      this, 'SlackSnsTopic', cdk.Fn.importValue('SlackSnsTopicArn')
    );

    const monitoring = new MonitoringFacade(this, 'ClassifierDashboard', {
      alarmFactoryDefaults: {
        actionsEnabled: true,
        alarmNamePrefix: 'Classifier-',
        action: new SnsAlarmActionStrategy({ onAlarmTopic: slackSnsTopic }),
        datapointsToAlarm: 1,
      },
    });

    monitoring
      .addLargeHeader('Gnome Classifier')
      .monitorStepFunction({
        stateMachine: props.stateMachine,
        humanReadableName: 'Contract Classifier',
        alarmFriendlyName: 'ContractClassifier',
        addFailedExecutionCountAlarm: {
          Critical: { maxErrorCount: 0 },
        },
      });
  }
}
