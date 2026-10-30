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
  batchFailures: Record<string, cloudwatch.Metric>;
  relationshipsPasses: cloudwatch.Metric;
  propositionExtractionFailures: cloudwatch.Metric;
  multiMarketSecurities: cloudwatch.Metric;
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

// Fires when an hourly log count stays below 1 for `hours` running, counting an hour with no lines at all: for a line
// that is logged on every pass, silence is the failure.
function silenceAlarm(
  metric: cloudwatch.Metric, alarmFriendlyName: string, description: string, tier: 'Error' | 'Warning', hours: number,
): CustomMetricWithAlarm {
  return {
    metric,
    alarmFriendlyName,
    addAlarm: {
      [tier]: {
        threshold: 1,
        comparisonOperator: cloudwatch.ComparisonOperator.LESS_THAN_THRESHOLD,
        treatMissingDataOverride: cloudwatch.TreatMissingData.BREACHING,
        evaluationPeriods: hours,
        datapointsToAlarm: hours,
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
          title: 'Failed batches per hour',
          // A queue worker logs a failed batch or idle pass and keeps polling, so restarts never show it; a few are a
          // flaky dependency, a steady stream is a pass that fails every time.
          metrics: Object.entries(props.batchFailures).map(([name, metric]) =>
            countAlarm(metric, `${name}BatchFailures`, 3,
              `${name} worker failed 3+ batches, idle passes or subject derivations in an hour`, 'Error')),
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
            // A failed request leaves its contracts unread until the next pass; one now and then is a timeout.
            countAlarm(props.propositionExtractionFailures, 'PropositionExtractionFailures', 5,
              'Claude proposition requests failed 5+ times in an hour (provider down, out of credit, bad answers)',
              'Error'),
            countAlarm(props.multiMarketSecurities, 'MultiMarketSecurities', 1,
              'A security\'s listings are different markets, so it gets no proposition until it is split', 'Warning'),
          ],
        },
        {
          title: 'Relationships passes per hour',
          // The worker logs a pass every minute it is idle and every batch, even with relationships off.
          metrics: [silenceAlarm(props.relationshipsPasses, 'RelationshipsSilent',
            'The relationships worker logged no pass for 2 hours: hung, or not running', 'Error', 2)],
        },
      ],
    });
  }
}
