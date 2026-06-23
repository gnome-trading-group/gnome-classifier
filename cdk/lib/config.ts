import { GnomeAccount, Stage } from "@gnome-trading-group/gnome-shared-cdk";

export const GITHUB_REPO = "gnome-trading-group/gnome-classifier";
export const GITHUB_BRANCH = "main";

export interface ClassifierConfig {
  account: GnomeAccount;
  slackChannel: string;
}

export const CONFIGS: { [stage in Stage]?: ClassifierConfig } = {
  [Stage.DEV]: { account: GnomeAccount.InfraDev, slackChannel: '' },
  [Stage.PROD]: { account: GnomeAccount.InfraProd, slackChannel: 'C0B60PCAPNC' },
};
