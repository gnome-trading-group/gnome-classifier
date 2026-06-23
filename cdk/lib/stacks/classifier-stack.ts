import * as cdk from 'aws-cdk-lib';
import { Construct } from 'constructs';
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

    const cacheBucket = new s3.Bucket(this, 'ClassifierCache', {
      bucketName: `gnome-classifier-cache-${props.stage}`,
      lifecycleRules: [{ expiration: cdk.Duration.days(90) }],
    });

    const registryEnvironment = {
      REGISTRY_API_URL: cdk.Fn.importValue('RegistryApiUrl'),
      REGISTRY_API_KEY_ID: cdk.Fn.importValue('RegistryApiKeyId'),
      ANTHROPIC_API_KEY_SECRET: 'anthropic-api-key',
      VOYAGE_API_KEY_SECRET: 'voyage-api-key',
      CACHE_BUCKET: cacheBucket.bucketName,
    };

    const imageAsset = join(__dirname, '..', '..', '..');

    // Constructed manually to avoid circular CFN dependency (state machine → Lambda → state machine)
    const stateMachineName = 'ContractClassifier';
    const stateMachineArn = `arn:aws:states:${this.region}:${this.account}:stateMachine:${stateMachineName}`;

    const fetchLambda = new lambda.DockerImageFunction(this, 'fetch-lambda', {
      code: lambda.DockerImageCode.fromImageAsset(imageAsset, {
        cmd: ['handler.fetch_and_create_entities'],
      }),
      timeout: cdk.Duration.minutes(10),
      memorySize: 1024,
      environment: { ...registryEnvironment, STATE_MACHINE_ARN: stateMachineArn },
    });

    const classifyLambda = new lambda.DockerImageFunction(this, 'classify-lambda', {
      code: lambda.DockerImageCode.fromImageAsset(imageAsset, {
        cmd: ['handler.classify_relationships_handler'],
      }),
      timeout: cdk.Duration.minutes(10),
      memorySize: 1024,
      environment: registryEnvironment,
    });

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

    for (const fn of [fetchLambda, classifyLambda]) {
      fn.addToRolePolicy(new iam.PolicyStatement({
        actions: ['apigateway:GET'],
        resources: [cdk.Fn.importValue('RegistryApiKeyArn')],
      }));
      anthropicApiKeySecret.grantRead(fn);
      voyageApiKeySecret.grantRead(fn);
      cacheBucket.grantReadWrite(fn);
    }

    fetchLambda.addToRolePolicy(new iam.PolicyStatement({
      actions: ['states:ListExecutions'],
      resources: [stateMachineArn],
    }));

    slackBotTokenSecret.grantRead(notifyLambda);

    const fetchTask = new tasks.LambdaInvoke(this, 'FetchAndCreateEntities', {
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

    const classifyTask = new tasks.LambdaInvoke(this, 'ClassifyRelationships', {
      lambdaFunction: classifyLambda,
      resultPath: '$.classification',
      retryOnServiceExceptions: true,
    });
    classifyTask.addRetry({
      errors: ['States.TaskFailed'],
      interval: cdk.Duration.seconds(30),
      maxAttempts: 2,
      backoffRate: 2,
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

    const definition = fetchTask.next(
      new sfn.Choice(this, 'HasNewEntities')
        .when(
          sfn.Condition.booleanEquals('$.has_new_entities', true),
          classifyTask.next(notifyTask),
        )
        .otherwise(new sfn.Succeed(this, 'NoNewEntities'))
    );

    this.stateMachine = new sfn.StateMachine(this, 'ClassifierStateMachine', {
      stateMachineName,
      definitionBody: sfn.DefinitionBody.fromChainable(definition),
      timeout: cdk.Duration.minutes(25),
    });

    const rule = new events.Rule(this, 'ClassifierRule', {
      schedule: events.Schedule.rate(cdk.Duration.minutes(5)),
      enabled: props.stage === Stage.PROD,
    });
    rule.addTarget(new targets.SfnStateMachine(this.stateMachine));
  }
}
