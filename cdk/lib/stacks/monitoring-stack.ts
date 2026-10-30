import * as cdk from 'aws-cdk-lib';
import * as cloudwatch from 'aws-cdk-lib/aws-cloudwatch';
import * as ecs from 'aws-cdk-lib/aws-ecs';
import * as sqs from 'aws-cdk-lib/aws-sqs';
import { Construct } from 'constructs';
import { CustomMetricWithAlarm, DefaultDashboardFactory, MonitoringFacade, noopAction } from 'cdk-monitoring-constructs';

interface Props extends cdk.StackProps {
  contractsQueue: sqs.Queue;
  contractsDlq: sqs.Queue;
  entitiesQueue: sqs.Queue;
  entitiesDlq: sqs.Queue;
  slackQueue: sqs.Queue;
  slackDlq: sqs.Queue;
  fetchService: ecs.Ec2Service;
  normalizeService: ecs.Ec2Service;
  relationshipsService: ecs.Ec2Service;
  notifyService: ecs.Ec2Service;
  workerStarts: Record<string, cloudwatch.Metric>;
  fetchCycleFailures: cloudwatch.Metric;
  venueFeedGaps: cloudwatch.Metric;
  closureChecksSkipped: cloudwatch.Metric;
  listingsReactivated: cloudwatch.Metric;
  identityRegressions: cloudwatch.Metric;
  venueReaderFallbacks: cloudwatch.Metric;
  propositionDrift: cloudwatch.Metric;
}

const AT_LEAST = cloudwatch.ComparisonOperator.GREATER_THAN_OR_EQUAL_TO_THRESHOLD;

// Fires when an hourly log count reaches `threshold`; an hour with no matching lines is healthy, not missing. The tier
// is the alert's severity (the registry's alerting reads it from the end of the alarm's name).
function countAlarm(
  metric: cloudwatch.Metric, alarmFriendlyName: string, threshold: number, description: string,
  tier: 'Error' | 'Warning', hoursRunning = 1,
): CustomMetricWithAlarm {
  return {
    metric,
    alarmFriendlyName,
    addAlarm: {
      [tier]: {
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

    // The alarms carry no action: the registry's alerting stack hears every alarm change through EventBridge. Nothing
    // here touches trading, so none of it pages.
    const monitoring = new MonitoringFacade(this, 'ClassifierDashboard', {
      // Only the alarms are used; each dashboard is billed monthly and nobody reads them.
      dashboardFactory: new DefaultDashboardFactory(this, 'ClassifierDashboardFactory', {
        dashboardNamePrefix: 'ClassifierDashboard',
        createDashboard: false,
      }),
      alarmFactoryDefaults: {
        actionsEnabled: true,
        alarmNamePrefix: 'Classifier-',
        action: noopAction(),
        datapointsToAlarm: 1,
      },
    });

    for (const [name, queue, dlq] of [
      ['Contracts', props.contractsQueue, props.contractsDlq],
      ['Entities', props.entitiesQueue, props.entitiesDlq],
      ['Slack', props.slackQueue, props.slackDlq],
    ] as [string, sqs.Queue, sqs.Queue][]) {
      monitoring.monitorSqsQueue({
        queue,
        humanReadableName: `${name} Queue`,
        alarmFriendlyName: name,
        addQueueMaxMessageAgeAlarm: {
          Warning: { maxAgeInSeconds: 3600 },
        },
      });
      monitoring.monitorSqsQueue({
        queue: dlq,
        humanReadableName: `${name} DLQ`,
        alarmFriendlyName: `${name}Dlq`,
        addQueueMaxSizeAlarm: {
          Error: { maxMessageCount: 1 },
        },
      });
    }

    for (const [name, service] of [
      ['Fetch', props.fetchService],
      ['Normalize', props.normalizeService],
      ['Relationships', props.relationshipsService],
      ['Notify', props.notifyService],
    ] as [string, ecs.Ec2Service][]) {
      monitoring.monitorSimpleEc2Service({
        ec2Service: service,
        humanReadableName: `${name} Worker`,
        alarmFriendlyName: `${name}Worker`,
        // ECS kills a container at its memory limit; this warns before the restarts begin.
        addMemoryUsageAlarm: {
          Warning: { maxUsagePercent: 90 },
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
            countAlarm(metric, `${name}WorkerRestarts`, 3, `${name} worker started 3+ times in an hour`, 'Error')),
        },
        {
          title: 'Fetch loop failures per hour',
          metrics: [countAlarm(props.fetchCycleFailures, 'FetchCycleFailures', 3,
            'Fetch-loop cycles (fetch, closure check, settle, reactivation, resolve, stale) or a venue settlement '
            + 'lookup failed 3+ times in an hour', 'Error')],
        },
        {
          title: 'Securities refused a second event per hour',
          // Any occurrence means an adapter changed the event it maps a market to, which splits markets.
          metrics: [countAlarm(props.identityRegressions, 'IdentityRegressions', 1,
            'An adapter tried to link an already-linked security to another event', 'Error')],
        },
        {
          title: 'Venue feed gaps per hour',
          // Stale cleanup runs hourly, so six hours running is a venue whose feed has been empty or partial for most
          // of the window after which its events would otherwise have been retired.
          metrics: [countAlarm(props.venueFeedGaps, 'VenueFeedGaps', 1,
            'A venue feed has been empty or partial for 6 hours', 'Warning', 6)],
        },
        {
          title: 'Venues misreporting markets per hour',
          // Either signature of a venue feed contradicting itself: hundreds of contracts vanishing from one fetch, or
          // a fetch listing dozens of markets as open that something had retired as closed (as on 2026-10-22).
          metrics: [
            countAlarm(props.closureChecksSkipped, 'ClosureChecksSkipped', 1,
              'Hundreds of contracts left a venue feed at once; they wait for the resolve cycle', 'Warning'),
            countAlarm(props.listingsReactivated, 'ListingsReactivated', 1,
              'A venue listed dozens of retired markets as open again; the classifier turned them back on', 'Warning'),
          ],
        },
        {
          title: 'Proposition readers per hour',
          // A venue rewording a template sends a whole family (every strike of a ladder) to the LLM at once; a few
          // misses are new market shapes. Any drift means a code reader and Claude read the same rules differently.
          metrics: [
            countAlarm(props.venueReaderFallbacks, 'VenueReaderFallbacks', 25,
              'A family the venue readers handle stopped parsing: its contracts now go to the LLM', 'Warning'),
            countAlarm(props.propositionDrift, 'PropositionDrift', 1,
              'The daily re-read by Claude disagreed with a venue reader on a field', 'Warning'),
          ],
        },
      ],
    });
  }
}
