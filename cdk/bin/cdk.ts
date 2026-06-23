#!/usr/bin/env node
import * as cdk from 'aws-cdk-lib';
import { ClassifierPipelineStack } from '../lib/classifier-pipeline-stack';
import { GnomeAccount } from '@gnome-trading-group/gnome-shared-cdk';

const app = new cdk.App();
new ClassifierPipelineStack(app, 'ClassifierPipelineStack', {
  env: GnomeAccount.InfraPipelines.environment,
});
app.synth();
