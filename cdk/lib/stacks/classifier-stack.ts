import * as cdk from 'aws-cdk-lib';
import { Construct } from 'constructs';
import * as autoscaling from 'aws-cdk-lib/aws-autoscaling';
import * as cloudwatch from 'aws-cdk-lib/aws-cloudwatch';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as ecs from 'aws-cdk-lib/aws-ecs';
import * as elasticache from 'aws-cdk-lib/aws-elasticache';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as logs from 'aws-cdk-lib/aws-logs';
import * as secrets from 'aws-cdk-lib/aws-secretsmanager';
import * as sns from 'aws-cdk-lib/aws-sns';
import * as snsSubscriptions from 'aws-cdk-lib/aws-sns-subscriptions';
import * as sqs from 'aws-cdk-lib/aws-sqs';
import { join } from 'path';
import { Stage } from '@gnome-trading-group/gnome-shared-cdk';

/** Namespace for metrics counted from the workers' logs. */
export const WORKER_METRICS_NAMESPACE = 'GnomeClassifier/Workers';

interface Props extends cdk.StackProps {
  stage: Stage;
  slackChannel: string;
}

export class ClassifierStack extends cdk.Stack {
  public readonly contractsQueue: sqs.Queue;
  public readonly contractsDlq: sqs.Queue;
  public readonly entitiesQueue: sqs.Queue;
  public readonly entitiesDlq: sqs.Queue;
  public readonly slackQueue: sqs.Queue;
  public readonly slackDlq: sqs.Queue;
  public readonly notificationsTopic: sns.Topic;
  public readonly fetchService: ecs.Ec2Service;
  public readonly normalizeService: ecs.Ec2Service;
  public readonly relationshipsService: ecs.Ec2Service;
  public readonly notifyService: ecs.Ec2Service;
  /** Hourly count of each worker's process starting, by worker name; more than a deploy's worth means it is crashing. */
  public readonly workerStarts: Record<string, cloudwatch.Metric> = {};
  /** Hourly count of failed batches, idle passes and subject derivations, by queue worker name; the worker survives
   * them and carries on. */
  public readonly batchFailures: Record<string, cloudwatch.Metric> = {};
  /** Hourly count of failed fetch-loop cycles (fetch, resolve, stale, settle) and venues failing a settle lookup. */
  public readonly fetchCycleFailures: cloudwatch.Metric;
  /** Hourly count of stale checks skipped because a venue's feed came back empty or partial. */
  public readonly venueFeedGaps: cloudwatch.Metric;
  /** Hourly count of fetches after which so many contracts left a venue's feed that none were looked up. */
  public readonly closureChecksSkipped: cloudwatch.Metric;
  /** Hourly count of fetches that turned back on many listings their venue had misreported as closed. */
  public readonly listingsReactivated: cloudwatch.Metric;
  /** Hourly count of securities an adapter tried to move to a second event. */
  public readonly identityRegressions: cloudwatch.Metric;
  /** Hourly count of contracts in a family the venue readers handle that their rules text no longer parsed. */
  public readonly venueReaderFallbacks: cloudwatch.Metric;
  /** Hourly count of fields where the daily re-read by Claude disagreed with a venue reader. */
  public readonly propositionDrift: cloudwatch.Metric;
  /** Hourly count of the relationships worker's passes (batch or idle, enabled or not); none means it hung. */
  public readonly relationshipsPasses: cloudwatch.Metric;
  /** Hourly count of Claude proposition requests that failed (provider down, out of credit, unreadable answer). */
  public readonly propositionExtractionFailures: cloudwatch.Metric;
  public readonly propositionCollisions: cloudwatch.Metric;
  /** Hourly count of securities whose listings carry different venue sources, so none is stored for them. */
  public readonly multiMarketSecurities: cloudwatch.Metric;

  constructor(scope: Construct, id: string, props: Props) {
    super(scope, id, props);

    const anthropicApiKeySecret = secrets.Secret.fromSecretNameV2(
      this, 'AnthropicApiKey', 'anthropic-api-key'
    );
    const voyageApiKeySecret = secrets.Secret.fromSecretNameV2(
      this, 'VoyageApiKey', 'voyage-api-key'
    );
    const slackBotTokenSecret = secrets.Secret.fromSecretNameV2(
      this, 'SlackBotToken', 'slack-bot-token'
    );
    const dbSecret = secrets.Secret.fromSecretNameV2(
      this, 'RegistryDbSecret', 'registry-database-root-user'
    );

    const vpc = ec2.Vpc.fromLookup(this, 'RegistryVpc', {
      vpcName: 'registry-database-vpc',
    });

    // ── SQS Queues ────────────────────────────────────────────────────

    this.contractsDlq = new sqs.Queue(this, 'ContractsDlq', {
      retentionPeriod: cdk.Duration.days(14),
    });
    this.contractsQueue = new sqs.Queue(this, 'ContractsQueue', {
      visibilityTimeout: cdk.Duration.minutes(5),
      deadLetterQueue: { queue: this.contractsDlq, maxReceiveCount: 3 },
    });

    this.entitiesDlq = new sqs.Queue(this, 'EntitiesDlq', {
      retentionPeriod: cdk.Duration.days(14),
    });
    this.entitiesQueue = new sqs.Queue(this, 'EntitiesQueue', {
      visibilityTimeout: cdk.Duration.minutes(30),
      deadLetterQueue: { queue: this.entitiesDlq, maxReceiveCount: 3 },
    });

    this.notificationsTopic = new sns.Topic(this, 'NotificationsTopic');

    this.slackDlq = new sqs.Queue(this, 'SlackDlq', {
      retentionPeriod: cdk.Duration.days(7),
    });
    this.slackQueue = new sqs.Queue(this, 'SlackQueue', {
      visibilityTimeout: cdk.Duration.minutes(2),
      deadLetterQueue: { queue: this.slackDlq, maxReceiveCount: 5 },
    });
    this.notificationsTopic.addSubscription(new snsSubscriptions.SqsSubscription(this.slackQueue));

    // ── ElastiCache Redis ─────────────────────────────────────────────

    const workerSg = new ec2.SecurityGroup(this, 'WorkerSg', {
      vpc,
      description: 'Classifier worker outbound access',
      allowAllOutbound: true,
    });

    const redisSubnetGroup = new elasticache.CfnSubnetGroup(this, 'RedisSubnetGroup', {
      description: 'Subnet group for classifier Redis',
      subnetIds: vpc.privateSubnets.map(s => s.subnetId),
    });

    const redisSg = new ec2.SecurityGroup(this, 'RedisSg', {
      vpc,
      description: 'ElastiCache Redis access',
    });
    redisSg.addIngressRule(workerSg, ec2.Port.tcp(6379), 'Allow workers to Redis');
    redisSg.addIngressRule(ec2.Peer.ipv4(vpc.vpcCidrBlock), ec2.Port.tcp(6379), 'Allow VPC to Redis for SSM tunnel');

    const redisCluster = new elasticache.CfnCacheCluster(this, 'RedisCluster', {
      cacheNodeType: 'cache.t3.micro',
      engine: 'redis',
      numCacheNodes: 1,
      cacheSubnetGroupName: redisSubnetGroup.ref,
      vpcSecurityGroupIds: [redisSg.securityGroupId],
    });

    const redisEndpoint = `redis://${redisCluster.attrRedisEndpointAddress}:${redisCluster.attrRedisEndpointPort}`;

    // ── ECS Cluster on EC2 (public subnet, spot) ──────────────────────

    const cluster = new ecs.Cluster(this, 'ClassifierCluster', { vpc });

    const workerAsg = new autoscaling.AutoScalingGroup(this, 'WorkerAsg', {
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.T3, ec2.InstanceSize.MEDIUM),
      machineImage: ecs.EcsOptimizedImage.amazonLinux2023(),
      vpc,
      vpcSubnets: { subnetType: ec2.SubnetType.PUBLIC },
      associatePublicIpAddress: true,
      spotPrice: '0.042',
      minCapacity: 1,
      maxCapacity: 2,
      securityGroup: workerSg,
    });

    const workerCapacity = new ecs.AsgCapacityProvider(this, 'WorkerCapacity', {
      autoScalingGroup: workerAsg,
      enableManagedTerminationProtection: false,
    });
    cluster.addAsgCapacityProvider(workerCapacity);

    // ── Shared environment ────────────────────────────────────────────

    const imageAsset = join(__dirname, '..', '..', '..');

    const controllerApiKeyId = cdk.Fn.importValue('ControllerServiceConfigApiKeyId');
    const controllerApiKeyArn = `arn:aws:apigateway:${this.region}::/apikeys/${controllerApiKeyId}`;
    const controllerEnv = {
      CONTROLLER_API_URL: cdk.Fn.importValue('ControllerApiUrl'),
      CONTROLLER_API_KEY_ID: controllerApiKeyId,
    };

    const sharedEnv = {
      REGISTRY_API_URL: cdk.Fn.importValue('RegistryApiUrl'),
      REGISTRY_API_KEY_ID: cdk.Fn.importValue('RegistryApiKeyId'),
      ANTHROPIC_API_KEY_SECRET: 'anthropic-api-key',
      VOYAGE_API_KEY_SECRET: 'voyage-api-key',
      REDIS_ENDPOINT: redisEndpoint,
      DB_SECRET_NAME: 'registry-database-root-user',
      CONTRACTS_QUEUE_URL: this.contractsQueue.queueUrl,
      ENTITIES_QUEUE_URL: this.entitiesQueue.queueUrl,
      NOTIFICATIONS_TOPIC_ARN: this.notificationsTopic.topicArn,
      SLACK_QUEUE_URL: this.slackQueue.queueUrl,
      ...controllerEnv,
    };

    // ── Helper: count matching log lines as a metric ──────────────────

    const logMetric = (logGroup: logs.ILogGroup, metricName: string, filterPattern: logs.IFilterPattern) => {
      new logs.MetricFilter(this, `${metricName}Filter`, {
        logGroup,
        metricNamespace: WORKER_METRICS_NAMESPACE,
        metricName,
        filterPattern,
        metricValue: '1',
        defaultValue: 0,
      });
      return new cloudwatch.Metric({
        namespace: WORKER_METRICS_NAMESPACE,
        metricName,
        statistic: 'Sum',
        period: cdk.Duration.hours(1),
      });
    };
    const workerLogGroups: Record<string, logs.ILogGroup> = {};

    // ── Helper: create a single-worker ECS service ────────────────────

    const createWorkerService = (
      id: string,
      workerCommand: string,
      environment: Record<string, string>,
      memoryLimitMiB: number,
      taskRoleGrants: (role: iam.Role) => void,
    ): ecs.Ec2Service => {
      const taskRole = new iam.Role(this, `${id}TaskRole`, {
        assumedBy: new iam.ServicePrincipal('ecs-tasks.amazonaws.com'),
      });
      taskRoleGrants(taskRole);

      const taskDef = new ecs.Ec2TaskDefinition(this, `${id}Task`, {
        taskRole,
      });

      const container = taskDef.addContainer(`${id}Container`, {
        image: ecs.ContainerImage.fromAsset(imageAsset),
        command: [workerCommand],
        memoryLimitMiB,
        environment,
        logging: ecs.LogDrivers.awsLogs({ streamPrefix: workerCommand }),
      });
      const logGroup = logs.LogGroup.fromLogGroupName(
        this, `${id}LogGroupRef`, container.logDriverConfig!.options!['awslogs-group'],
      );
      workerLogGroups[id] = logGroup;
      // Every worker logs "<Id>Worker started" when its process starts.
      this.workerStarts[id] = logMetric(logGroup, `${id}WorkerStarts`, logs.FilterPattern.literal(`"${id}Worker started"`));
      if (id !== 'Fetch') {
        this.batchFailures[id] = logMetric(logGroup, `${id}BatchFailures`,
          logs.FilterPattern.anyTerm('process_batch failed', 'on_idle failed', 'subject derivation failed',
            'relationship reduction mismatch'));
      }

      return new ecs.Ec2Service(this, `${id}Service`, {
        cluster,
        taskDefinition: taskDef,
        desiredCount: 1,
        minHealthyPercent: 0,
        maxHealthyPercent: 100,
        capacityProviderStrategies: [{ capacityProvider: workerCapacity.capacityProviderName, weight: 1 }],
      });
    };

    // ── FetchRunner (ECS long-running service) ────────────────────────

    this.fetchService = createWorkerService('Fetch', 'fetch', {
      REGISTRY_API_URL: cdk.Fn.importValue('RegistryApiUrl'),
      REGISTRY_API_KEY_ID: cdk.Fn.importValue('RegistryApiKeyId'),
      REDIS_ENDPOINT: redisEndpoint,
      CONTRACTS_QUEUE_URL: this.contractsQueue.queueUrl,
      // The settle cycle reads which deactivated outcomes still lack a settlement value.
      DB_SECRET_NAME: 'registry-database-root-user',
      // Events resolve on their settlement values, recorded here, and are announced to the launcher and Slack.
      NOTIFICATIONS_TOPIC_ARN: this.notificationsTopic.topicArn,
      ...controllerEnv,
    // A full fetch holds every venue's markets at once (Polymarket International alone is ~460k); at 1024 MiB its
    // peaks were OOM-killing the worker every few minutes, cutting off the resolve and stale cycles behind it.
    }, 2048, (role) => {
      this.contractsQueue.grantSendMessages(role);
      this.notificationsTopic.grantPublish(role);
      dbSecret.grantRead(role);
      role.addToPolicy(new iam.PolicyStatement({
        actions: ['apigateway:GET'],
        resources: [cdk.Fn.importValue('RegistryApiKeyArn'), controllerApiKeyArn],
      }));
    });

    this.fetchCycleFailures = logMetric(workerLogGroups.Fetch, 'FetchCycleFailures',
      logs.FilterPattern.anyTerm('cycle failed', 'Failed to fetch settlements from'));
    this.venueFeedGaps = logMetric(workerLogGroups.Fetch, 'VenueFeedGaps',
      logs.FilterPattern.literal('"not counting misses this cycle"'));
    this.closureChecksSkipped = logMetric(workerLogGroups.Fetch, 'ClosureChecksSkipped',
      logs.FilterPattern.literal('"not checking them this cycle"'));
    this.listingsReactivated = logMetric(workerLogGroups.Fetch, 'ListingsReactivated',
      logs.FilterPattern.literal('"retired listings as open again"'));

    // ── NormalizeWorker ───────────────────────────────────────────────

    this.normalizeService = createWorkerService('Normalize', 'normalize', {
      ...sharedEnv,
      SLACK_CHANNEL: props.slackChannel,
    }, 512, (role) => {
      this.contractsQueue.grantConsumeMessages(role);
      this.entitiesQueue.grantSendMessages(role);
      this.notificationsTopic.grantPublish(role);
      dbSecret.grantRead(role);
      role.addToPolicy(new iam.PolicyStatement({
        actions: ['apigateway:GET'],
        resources: [cdk.Fn.importValue('RegistryApiKeyArn'), controllerApiKeyArn],
      }));
      // The canonicalize model is runtime config, and cross-region profiles route to foundation models in other
      // regions (global profiles to a region-less ARN), so neither resource can be pinned at deploy time.
      role.addToPolicy(new iam.PolicyStatement({
        actions: ['bedrock:InvokeModel'],
        resources: [
          `arn:aws:bedrock:${this.region}:${this.account}:inference-profile/*`,
          'arn:aws:bedrock:*::foundation-model/*',
        ],
      }));
    });

    // Entities are created by the normalize worker, which logs when it refuses to link a security a second time.
    this.identityRegressions = logMetric(workerLogGroups.Normalize, 'IdentityRegressions',
      logs.FilterPattern.literal('"already in event"'));

    // ── RelationshipsWorker ───────────────────────────────────────────

    this.relationshipsService = createWorkerService('Relationships', 'relationships', sharedEnv, 512, (role) => {
      this.entitiesQueue.grantConsumeMessages(role);
      this.notificationsTopic.grantPublish(role);
      anthropicApiKeySecret.grantRead(role);
      voyageApiKeySecret.grantRead(role);
      dbSecret.grantRead(role);
      role.addToPolicy(new iam.PolicyStatement({
        actions: ['apigateway:GET'],
        resources: [cdk.Fn.importValue('RegistryApiKeyArn'), controllerApiKeyArn],
      }));
    });

    // Propositions are extracted by the relationships worker, which logs each contract a venue reader should have read
    // but could not (a venue rewording its rules) and each field where the daily re-read by Claude disagreed.
    this.venueReaderFallbacks = logMetric(workerLogGroups.Relationships, 'VenueReaderFallbacks',
      logs.FilterPattern.literal('"venue reader fell back"'));
    this.propositionDrift = logMetric(workerLogGroups.Relationships, 'PropositionDrift',
      logs.FilterPattern.literal('"proposition drift"'));
    this.relationshipsPasses = logMetric(workerLogGroups.Relationships, 'RelationshipsPasses',
      logs.FilterPattern.literal('"Relationships pass"'));
    this.propositionExtractionFailures = logMetric(workerLogGroups.Relationships, 'PropositionExtractionFailures',
      logs.FilterPattern.literal('"Proposition request for"'));
    this.propositionCollisions = logMetric(workerLogGroups.Relationships, 'PropositionCollisions',
      logs.FilterPattern.literal('"proposition collision"'));
    this.multiMarketSecurities = logMetric(workerLogGroups.Normalize, 'MultiMarketSecurities',
      logs.FilterPattern.literal('"listings with different sources"'));

    // ── NotifyWorker ──────────────────────────────────────────────────

    this.notifyService = createWorkerService('Notify', 'notify', {
      SLACK_QUEUE_URL: this.slackQueue.queueUrl,
      SLACK_CHANNEL: props.slackChannel,
      SLACK_BOT_TOKEN_SECRET: 'slack-bot-token',
      ...controllerEnv,
    }, 128, (role) => {
      this.slackQueue.grantConsumeMessages(role);
      slackBotTokenSecret.grantRead(role);
      role.addToPolicy(new iam.PolicyStatement({
        actions: ['apigateway:GET'],
        resources: [controllerApiKeyArn],
      }));
    });

    new cdk.CfnOutput(this, 'RedisEndpoint', {
      value: redisEndpoint,
      description: 'ElastiCache Redis endpoint for SSM tunnel',
    });

    new cdk.CfnOutput(this, 'NotificationsTopicArn', {
      value: this.notificationsTopic.topicArn,
      exportName: 'ClassifierNotificationsTopicArn',
    });
  }
}
