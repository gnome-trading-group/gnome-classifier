import * as cdk from 'aws-cdk-lib';
import { Construct } from 'constructs';
import * as autoscaling from 'aws-cdk-lib/aws-autoscaling';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as ecs from 'aws-cdk-lib/aws-ecs';
import * as elasticache from 'aws-cdk-lib/aws-elasticache';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as secrets from 'aws-cdk-lib/aws-secretsmanager';
import * as sns from 'aws-cdk-lib/aws-sns';
import * as snsSubscriptions from 'aws-cdk-lib/aws-sns-subscriptions';
import * as sqs from 'aws-cdk-lib/aws-sqs';
import { join } from 'path';
import { Stage } from '@gnome-trading-group/gnome-shared-cdk';

interface Props extends cdk.StackProps {
  stage: Stage;
  slackChannel: string;
}

export class ClassifierStack extends cdk.Stack {
  public readonly contractsQueue: sqs.Queue;
  public readonly contractsDlq: sqs.Queue;
  public readonly entitiesQueue: sqs.Queue;
  public readonly entitiesDlq: sqs.Queue;
  public readonly embeddingsQueue: sqs.Queue;
  public readonly embeddingsDlq: sqs.Queue;
  public readonly slackQueue: sqs.Queue;
  public readonly slackDlq: sqs.Queue;
  public readonly notificationsTopic: sns.Topic;
  public readonly fetchService: ecs.Ec2Service;
  public readonly normalizeService: ecs.Ec2Service;
  public readonly embedService: ecs.Ec2Service;
  public readonly relationshipsService: ecs.Ec2Service;
  public readonly notifyService: ecs.Ec2Service;

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

    const cacheBucket = new s3.Bucket(this, 'ClassifierCache', {
      bucketName: `gnome-classifier-cache-${props.stage}`,
      lifecycleRules: [{ expiration: cdk.Duration.days(90) }],
    });

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

    this.embeddingsDlq = new sqs.Queue(this, 'EmbeddingsDlq', {
      retentionPeriod: cdk.Duration.days(14),
    });
    this.embeddingsQueue = new sqs.Queue(this, 'EmbeddingsQueue', {
      visibilityTimeout: cdk.Duration.minutes(15),
      deadLetterQueue: { queue: this.embeddingsDlq, maxReceiveCount: 3 },
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
      CACHE_BUCKET: cacheBucket.bucketName,
      REDIS_ENDPOINT: redisEndpoint,
      DB_SECRET_NAME: 'registry-database-root-user',
      CONTRACTS_QUEUE_URL: this.contractsQueue.queueUrl,
      ENTITIES_QUEUE_URL: this.entitiesQueue.queueUrl,
      EMBEDDINGS_QUEUE_URL: this.embeddingsQueue.queueUrl,
      NOTIFICATIONS_TOPIC_ARN: this.notificationsTopic.topicArn,
      SLACK_QUEUE_URL: this.slackQueue.queueUrl,
      ...controllerEnv,
    };

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

      taskDef.addContainer(`${id}Container`, {
        image: ecs.ContainerImage.fromAsset(imageAsset),
        command: [workerCommand],
        memoryLimitMiB,
        environment,
        logging: ecs.LogDrivers.awsLogs({ streamPrefix: workerCommand }),
      });

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
      ...controllerEnv,
    }, 512, (role) => {
      this.contractsQueue.grantSendMessages(role);
      role.addToPolicy(new iam.PolicyStatement({
        actions: ['apigateway:GET'],
        resources: [cdk.Fn.importValue('RegistryApiKeyArn'), controllerApiKeyArn],
      }));
    });

    // ── NormalizeWorker ───────────────────────────────────────────────

    this.normalizeService = createWorkerService('Normalize', 'normalize', {
      ...sharedEnv,
      SLACK_CHANNEL: props.slackChannel,
    }, 512, (role) => {
      this.contractsQueue.grantConsumeMessages(role);
      this.entitiesQueue.grantSendMessages(role);
      this.notificationsTopic.grantPublish(role);
      anthropicApiKeySecret.grantRead(role);
      dbSecret.grantRead(role);
      cacheBucket.grantReadWrite(role);
      role.addToPolicy(new iam.PolicyStatement({
        actions: ['apigateway:GET'],
        resources: [cdk.Fn.importValue('RegistryApiKeyArn'), controllerApiKeyArn],
      }));
    });

    // ── EmbedWorker ───────────────────────────────────────────────────

    this.embedService = createWorkerService('Embed', 'embed', {
      VOYAGE_API_KEY_SECRET: 'voyage-api-key',
      DB_SECRET_NAME: 'registry-database-root-user',
      ENTITIES_QUEUE_URL: this.entitiesQueue.queueUrl,
      EMBEDDINGS_QUEUE_URL: this.embeddingsQueue.queueUrl,
      ...controllerEnv,
    }, 512, (role) => {
      this.entitiesQueue.grantConsumeMessages(role);
      this.embeddingsQueue.grantSendMessages(role);
      voyageApiKeySecret.grantRead(role);
      dbSecret.grantRead(role);
      role.addToPolicy(new iam.PolicyStatement({
        actions: ['apigateway:GET'],
        resources: [controllerApiKeyArn],
      }));
    });

    // ── RelationshipsWorker ───────────────────────────────────────────

    this.relationshipsService = createWorkerService('Relationships', 'relationships', sharedEnv, 512, (role) => {
      this.embeddingsQueue.grantConsumeMessages(role);
      this.notificationsTopic.grantPublish(role);
      anthropicApiKeySecret.grantRead(role);
      voyageApiKeySecret.grantRead(role);
      dbSecret.grantRead(role);
      cacheBucket.grantReadWrite(role);
      role.addToPolicy(new iam.PolicyStatement({
        actions: ['apigateway:GET'],
        resources: [cdk.Fn.importValue('RegistryApiKeyArn'), controllerApiKeyArn],
      }));
    });

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
  }
}
