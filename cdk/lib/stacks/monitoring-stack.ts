import * as cdk from 'aws-cdk-lib';
import * as cloudwatch from 'aws-cdk-lib/aws-cloudwatch';
import * as ecs from 'aws-cdk-lib/aws-ecs';
import * as sns from 'aws-cdk-lib/aws-sns';
import * as sqs from 'aws-cdk-lib/aws-sqs';
import { Construct } from 'constructs';
import { CustomMetricWithAlarm, MonitoringFacade, SnsAlarmActionStrategy } from 'cdk-monitoring-constructs';

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
  workerStarts: Record<string, cloudwatch.Metric>;
  fetchCycleFailures: cloudwatch.Metric;
  venueFeedGaps: cloudwatch.Metric;
  identityRegressions: cloudwatch.Metric;
}

const AT_LEAST = cloudwatch.ComparisonOperator.GREATER_THAN_OR_EQUAL_TO_THRESHOLD;

// Fires when an hourly log count reaches `threshold`; an hour with no matching lines is healthy, not missing.
function countAlarm(
  metric: cloudwatch.Metric, alarmFriendlyName: string, threshold: number, description: string, hoursRunning = 1,
): CustomMetricWithAlarm {
  return {
    metric,
    alarmFriendlyName,
    addAlarm: {
      Critical: {
        threshold,
        comparisonOperator: AT_LEAST,
        treatMissingDataOverride: cloudwatch.TreatMissingData.NOT_BREACHING,
        evaluationPeriods: hoursRunning,
        datapointsToAlarm: hoursRunning,
        additionalDescription: description,
      },
    },
  };
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
        // ECS kills a container at its memory limit; this warns before the restarts begin.
        addMemoryUsageAlarm: {
          Critical: { maxUsagePercent: 90 },
        },
      });
    }

    monitoring.monitorCustom({
      humanReadableName: 'Worker Health',
      alarmFriendlyName: 'WorkerHealth',
      metricGroups: [
        {
          title: 'Worker starts per hour',
          // A deploy starts each worker once; three in an hour means it keeps dying (e.g. out of memory).
          metrics: Object.entries(props.workerStarts).map(([name, metric]) =>
            countAlarm(metric, `${name}WorkerRestarts`, 3, `${name} worker started 3+ times in an hour`)),
        },
        {
          title: 'Fetch loop failures per hour',
          metrics: [countAlarm(props.fetchCycleFailures, 'FetchCycleFailures', 3,
            'Fetch, resolve, stale or settle cycles (or a venue settlement lookup) failed 3+ times in an hour')],
        },
        {
          title: 'Securities refused a second event per hour',
          // Any occurrence means an adapter changed the event it maps a market to, which splits markets.
          metrics: [countAlarm(props.identityRegressions, 'IdentityRegressions', 1,
            'An adapter tried to link an already-linked security to another event')],
        },
        {
          title: 'Venue feed gaps per hour',
          // Stale cleanup runs hourly, so six hours running is a venue whose feed has been empty or partial for most
          // of the window after which its events would otherwise have been retired.
          metrics: [countAlarm(props.venueFeedGaps, 'VenueFeedGaps', 1,
            'A venue feed has been empty or partial for 6 hours', 6)],
        },
      ],
    });
  }
}
