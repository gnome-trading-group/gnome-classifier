import * as cdk from 'aws-cdk-lib';
import { Construct } from 'constructs';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as elasticache from 'aws-cdk-lib/aws-elasticache';
import * as events from 'aws-cdk-lib/aws-events';
import * as targets from 'aws-cdk-lib/aws-events-targets';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as secrets from 'aws-cdk-lib/aws-secretsmanager';
import * as sfn from 'aws-cdk-lib/aws-stepfunctions';
import * as tasks from 'aws-cdk-lib/aws-stepfunctions-tasks';
import { join } from 'path';
import { Stage } from '@gnome-trading-group/gnome-shared-cdk';

interface Props extends cdk.StackProps {
  stage: Stage;
  slackChannel: string;
}

export class ClassifierStack extends cdk.Stack {
  public readonly stateMachine: sfn.StateMachine;

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

    // Import the registry VPC so we can place Lambdas alongside RDS/Redis
    const vpc = ec2.Vpc.fromLookup(this, 'RegistryVpc', {
      vpcName: 'registry-database-vpc',
    });

    // Security group for classifier Lambdas
    const lambdaSg = new ec2.SecurityGroup(this, 'ClassifierLambdaSg', {
      vpc,
      description: 'Classifier Lambda outbound access',
      allowAllOutbound: true,
    });

    // ElastiCache Redis subnet group (reuses the VPC's existing private subnets)
    const redisSubnetGroup = new elasticache.CfnSubnetGroup(this, 'RedisSubnetGroup', {
      description: 'Subnet group for classifier Redis',
      subnetIds: vpc.privateSubnets.map(s => s.subnetId),
    });

    const redisSg = new ec2.SecurityGroup(this, 'RedisSg', {
      vpc,
      description: 'ElastiCache Redis access',
    });
    redisSg.addIngressRule(lambdaSg, ec2.Port.tcp(6379), 'Allow Lambda to Redis');
    redisSg.addIngressRule(ec2.Peer.ipv4(vpc.vpcCidrBlock), ec2.Port.tcp(6379), 'Allow VPC to Redis for SSM tunnel');

    const redisCluster = new elasticache.CfnCacheCluster(this, 'RedisCluster', {
      cacheNodeType: 'cache.t3.small',
      engine: 'redis',
      numCacheNodes: 1,
      cacheSubnetGroupName: redisSubnetGroup.ref,
      vpcSecurityGroupIds: [redisSg.securityGroupId],
    });

    const redisEndpoint = `redis://${redisCluster.attrRedisEndpointAddress}:${redisCluster.attrRedisEndpointPort}`;

    const registryEnvironment = {
      REGISTRY_API_URL: cdk.Fn.importValue('RegistryApiUrl'),
      REGISTRY_API_KEY_ID: cdk.Fn.importValue('RegistryApiKeyId'),
      ANTHROPIC_API_KEY_SECRET: 'anthropic-api-key',
      VOYAGE_API_KEY_SECRET: 'voyage-api-key',
      CACHE_BUCKET: cacheBucket.bucketName,
      REDIS_ENDPOINT: redisEndpoint,
      DB_SECRET_NAME: 'registry-database-root-user',
    };

    const imageAsset = join(__dirname, '..', '..', '..');

    // Constructed manually to avoid circular CFN dependency (state machine → Lambda → state machine)
    const stateMachineName = 'ContractClassifier';
    const stateMachineArn = `arn:aws:states:${this.region}:${this.account}:stateMachine:${stateMachineName}`;

    const vpcConfig = {
      vpc,
      vpcSubnets: { subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS },
      securityGroups: [lambdaSg],
    };

    const fetchLambda = new lambda.DockerImageFunction(this, 'fetch-lambda', {
      code: lambda.DockerImageCode.fromImageAsset(imageAsset, {
        cmd: ['handler.fetch_and_prepare'],
      }),
      timeout: cdk.Duration.minutes(10),
      memorySize: 2048,
      environment: { ...registryEnvironment, STATE_MACHINE_ARN: stateMachineArn },
      ...vpcConfig,
    });

    const submitCanonLambda = new lambda.DockerImageFunction(this, 'submit-canon-lambda', {
      code: lambda.DockerImageCode.fromImageAsset(imageAsset, {
        cmd: ['handler.submit_canon_handler'],
      }),
      timeout: cdk.Duration.minutes(5),
      memorySize: 2048,
      environment: registryEnvironment,
      ...vpcConfig,
    });

    // Generic check-batch lambda reused for both canon and semantic poll loops
    const checkBatchLambda = new lambda.DockerImageFunction(this, 'check-batch-lambda', {
      code: lambda.DockerImageCode.fromImageAsset(imageAsset, {
        cmd: ['handler.check_batch_handler'],
      }),
      timeout: cdk.Duration.seconds(30),
      memorySize: 256,
      environment: registryEnvironment,
      ...vpcConfig,
    });

    const collectCanonLambda = new lambda.DockerImageFunction(this, 'collect-canon-lambda', {
      code: lambda.DockerImageCode.fromImageAsset(imageAsset, {
        cmd: ['handler.collect_canon_handler'],
      }),
      timeout: cdk.Duration.minutes(10),
      memorySize: 2048,
      environment: registryEnvironment,
      ...vpcConfig,
    });

    const classifyStructuralLambda = new lambda.DockerImageFunction(this, 'classify-structural-lambda', {
      code: lambda.DockerImageCode.fromImageAsset(imageAsset, {
        cmd: ['handler.classify_structural_handler'],
      }),
      timeout: cdk.Duration.minutes(5),
      memorySize: 2048,
      environment: registryEnvironment,
      ...vpcConfig,
    });

    const submitSemanticLambda = new lambda.DockerImageFunction(this, 'submit-semantic-lambda', {
      code: lambda.DockerImageCode.fromImageAsset(imageAsset, {
        cmd: ['handler.submit_semantic_batch_handler'],
      }),
      timeout: cdk.Duration.minutes(2),
      memorySize: 2048,
      environment: registryEnvironment,
      ...vpcConfig,
    });

    const collectSemanticLambda = new lambda.DockerImageFunction(this, 'collect-semantic-lambda', {
      code: lambda.DockerImageCode.fromImageAsset(imageAsset, {
        cmd: ['handler.collect_semantic_results_handler'],
      }),
      timeout: cdk.Duration.minutes(5),
      memorySize: 2048,
      environment: registryEnvironment,
      ...vpcConfig,
    });

    // notify-lambda stays outside VPC — only calls Slack, no DB/Redis needed
    const notifyLambda = new lambda.DockerImageFunction(this, 'notify-lambda', {
      code: lambda.DockerImageCode.fromImageAsset(imageAsset, {
        cmd: ['handler.send_notification'],
      }),
      timeout: cdk.Duration.seconds(30),
      memorySize: 256,
      environment: {
        SLACK_BOT_TOKEN_SECRET: 'slack-bot-token',
        SLACK_CHANNEL: props.slackChannel,
      },
    });

    const classifyLambdas = [
      submitCanonLambda, checkBatchLambda, collectCanonLambda,
      classifyStructuralLambda, submitSemanticLambda, collectSemanticLambda,
    ];

    for (const fn of [fetchLambda, ...classifyLambdas]) {
      fn.addToRolePolicy(new iam.PolicyStatement({
        actions: ['apigateway:GET'],
        resources: [cdk.Fn.importValue('RegistryApiKeyArn')],
      }));
      anthropicApiKeySecret.grantRead(fn);
      voyageApiKeySecret.grantRead(fn);
      dbSecret.grantRead(fn);
      cacheBucket.grantReadWrite(fn);
    }

    fetchLambda.addToRolePolicy(new iam.PolicyStatement({
      actions: ['states:ListExecutions'],
      resources: [stateMachineArn],
    }));

    slackBotTokenSecret.grantRead(notifyLambda);

    // ── Step Functions tasks ──────────────────────────────────────────────────

    const fetchTask = new tasks.LambdaInvoke(this, 'FetchAndPrepare', {
      lambdaFunction: fetchLambda,
      outputPath: '$.Payload',
      retryOnServiceExceptions: true,
    });
    fetchTask.addRetry({
      errors: ['States.TaskFailed'],
      interval: cdk.Duration.seconds(30),
      maxAttempts: 2,
      backoffRate: 2,
    });

    const submitCanonTask = new tasks.LambdaInvoke(this, 'SubmitCanon', {
      lambdaFunction: submitCanonLambda,
      outputPath: '$.Payload',
      retryOnServiceExceptions: true,
    });
    submitCanonTask.addRetry({
      errors: ['States.TaskFailed'],
      interval: cdk.Duration.seconds(30),
      maxAttempts: 2,
      backoffRate: 2,
    });

    const waitForCanon = new sfn.Wait(this, 'WaitForCanon', {
      time: sfn.WaitTime.duration(cdk.Duration.seconds(30)),
    });

    const checkCanonBatchTask = new tasks.LambdaInvoke(this, 'CheckCanonBatch', {
      lambdaFunction: checkBatchLambda,
      outputPath: '$.Payload',
      retryOnServiceExceptions: true,
    });

    const collectCanonTask = new tasks.LambdaInvoke(this, 'CollectCanon', {
      lambdaFunction: collectCanonLambda,
      outputPath: '$.Payload',
      retryOnServiceExceptions: true,
    });
    collectCanonTask.addRetry({
      errors: ['States.TaskFailed'],
      interval: cdk.Duration.seconds(30),
      maxAttempts: 2,
      backoffRate: 2,
    });

    const classifyStructuralTask = new tasks.LambdaInvoke(this, 'ClassifyStructural', {
      lambdaFunction: classifyStructuralLambda,
      outputPath: '$.Payload',
      retryOnServiceExceptions: true,
    });
    classifyStructuralTask.addRetry({
      errors: ['States.TaskFailed'],
      interval: cdk.Duration.seconds(30),
      maxAttempts: 2,
      backoffRate: 2,
    });

    const submitSemanticTask = new tasks.LambdaInvoke(this, 'SubmitSemanticBatch', {
      lambdaFunction: submitSemanticLambda,
      outputPath: '$.Payload',
      retryOnServiceExceptions: true,
    });

    const waitForSemantic = new sfn.Wait(this, 'WaitForSemantic', {
      time: sfn.WaitTime.duration(cdk.Duration.seconds(30)),
    });

    const checkSemanticBatchTask = new tasks.LambdaInvoke(this, 'CheckSemanticBatch', {
      lambdaFunction: checkBatchLambda,
      outputPath: '$.Payload',
      retryOnServiceExceptions: true,
    });

    const collectSemanticTask = new tasks.LambdaInvoke(this, 'CollectSemanticResults', {
      lambdaFunction: collectSemanticLambda,
      resultPath: '$.semantic_result',
      retryOnServiceExceptions: true,
    });

    const notifyTask = new tasks.LambdaInvoke(this, 'SendNotification', {
      lambdaFunction: notifyLambda,
      outputPath: '$.Payload',
      retryOnServiceExceptions: true,
    });
    notifyTask.addRetry({
      errors: ['States.TaskFailed'],
      interval: cdk.Duration.seconds(10),
      maxAttempts: 2,
      backoffRate: 2,
    });

    // Semantic poll loop: wait → check → done or loop back
    const semanticPollLoop = waitForSemantic
      .next(checkSemanticBatchTask)
      .next(
        new sfn.Choice(this, 'IsSemanticBatchComplete')
          .when(
            sfn.Condition.booleanEquals('$.batch_complete', true),
            collectSemanticTask.next(notifyTask),
          )
          .otherwise(waitForSemantic)
      );

    // Semantic branch: submit → if sync complete go to notify, else enter poll loop
    const semanticFlow = submitSemanticTask.next(
      new sfn.Choice(this, 'IsSemanticSyncComplete')
        .when(
          sfn.Condition.booleanEquals('$.batch_complete', true),
          notifyTask,
        )
        .otherwise(semanticPollLoop)
    );

    // Classification flow: structural → optionally semantic
    const classificationFlow = classifyStructuralTask.next(
      new sfn.Choice(this, 'NeedsSemantic')
        .when(
          sfn.Condition.booleanEquals('$.needs_semantic', true),
          semanticFlow,
        )
        .otherwise(notifyTask)
    );

    // After entity creation (both sync and batch canon paths), gate on has_new_entities
    const hasNewEntitiesChoice = new sfn.Choice(this, 'HasNewEntities')
      .when(
        sfn.Condition.booleanEquals('$.has_new_entities', true),
        classificationFlow,
      )
      .otherwise(notifyTask);

    // Canon batch poll loop: wait → check → done or loop back → collect → HasNewEntities
    const canonBatchPath = collectCanonTask.next(hasNewEntitiesChoice);
    const canonPollLoop = waitForCanon
      .next(checkCanonBatchTask)
      .next(
        new sfn.Choice(this, 'IsCanonBatchComplete')
          .when(
            sfn.Condition.booleanEquals('$.batch_complete', true),
            canonBatchPath,
          )
          .otherwise(waitForCanon)
      );

    // Main definition
    const definition = fetchTask.next(
      new sfn.Choice(this, 'HasNewContracts')
        .when(
          sfn.Condition.booleanEquals('$.has_new_contracts', true),
          submitCanonTask.next(
            new sfn.Choice(this, 'IsCanonSyncComplete')
              .when(
                sfn.Condition.booleanEquals('$.batch_complete', true),
                hasNewEntitiesChoice,
              )
              .otherwise(canonPollLoop)
          )
        )
        .otherwise(new sfn.Succeed(this, 'NoNewContracts'))
    );

    this.stateMachine = new sfn.StateMachine(this, 'ClassifierStateMachine', {
      stateMachineName,
      definitionBody: sfn.DefinitionBody.fromChainable(definition),
      timeout: cdk.Duration.minutes(45),
    });

    const rule = new events.Rule(this, 'ClassifierRule', {
      schedule: events.Schedule.rate(cdk.Duration.minutes(30)),
      enabled: false, // temporarily disabled during bootstrap
    });
    rule.addTarget(new targets.SfnStateMachine(this.stateMachine));

    new cdk.CfnOutput(this, 'RedisEndpoint', {
      value: redisEndpoint,
      description: 'ElastiCache Redis endpoint for SSM tunnel',
    });
  }
}
