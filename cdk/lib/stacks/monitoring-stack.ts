import * as cdk from 'aws-cdk-lib';
import * as ecs from 'aws-cdk-lib/aws-ecs';
import * as sns from 'aws-cdk-lib/aws-sns';
import * as sqs from 'aws-cdk-lib/aws-sqs';
import { Construct } from 'constructs';
import { MonitoringFacade, SnsAlarmActionStrategy } from 'cdk-monitoring-constructs';

interface Props extends cdk.StackProps {
  contractsQueue: sqs.Queue;
  contractsDlq: sqs.Queue;
  entitiesQueue: sqs.Queue;
  entitiesDlq: sqs.Queue;
  embeddingsQueue: sqs.Queue;
  embeddingsDlq: sqs.Queue;
  slackQueue: sqs.Queue;
  slackDlq: sqs.Queue;
  fetchService: ecs.Ec2Service;
  normalizeService: ecs.Ec2Service;
  embedService: ecs.Ec2Service;
  relationshipsService: ecs.Ec2Service;
  notifyService: ecs.Ec2Service;
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

    monitoring.addLargeHeader('Gnome Classifier');

    for (const [name, queue, dlq] of [
      ['Contracts', props.contractsQueue, props.contractsDlq],
      ['Entities', props.entitiesQueue, props.entitiesDlq],
      ['Embeddings', props.embeddingsQueue, props.embeddingsDlq],
      ['Slack', props.slackQueue, props.slackDlq],
    ] as [string, sqs.Queue, sqs.Queue][]) {
      monitoring.monitorSqsQueue({
        queue,
        humanReadableName: `${name} Queue`,
        alarmFriendlyName: name,
        addQueueMaxMessageAgeAlarm: {
          Critical: { maxAgeInSeconds: 3600 },
        },
      });
      monitoring.monitorSqsQueue({
        queue: dlq,
        humanReadableName: `${name} DLQ`,
        alarmFriendlyName: `${name}Dlq`,
        addQueueMaxSizeAlarm: {
          Critical: { maxMessageCount: 1 },
        },
      });
    }

    for (const [name, service] of [
      ['Fetch', props.fetchService],
      ['Normalize', props.normalizeService],
      ['Embed', props.embedService],
      ['Relationships', props.relationshipsService],
      ['Notify', props.notifyService],
    ] as [string, ecs.Ec2Service][]) {
      monitoring.monitorSimpleEc2Service({
        ec2Service: service,
        humanReadableName: `${name} Worker`,
        alarmFriendlyName: `${name}Worker`,
      });
    }
  }
}
