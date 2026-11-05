import click

from classifier.utils import setup_logging
from classifier.workers.config import WorkerConfig, init_db, init_registry
from classifier.workers.fetch import FetchRunner
from classifier.workers.normalize import NormalizeWorker
from classifier.workers.notify import NotifyWorker
from classifier.workers.relationships import RelationshipsWorker
from classifier.workers.repair_overdue import repair_overdue
from classifier.workers.split_securities import split_securities
from classifier.workers.sync_names import sync_names


@click.group()
def cli():
    setup_logging()


@cli.command()
def normalize():
    NormalizeWorker(WorkerConfig()).run()


@cli.command()
def relationships():
    RelationshipsWorker(WorkerConfig()).run()


@cli.command()
def notify():
    NotifyWorker(WorkerConfig()).run()


@cli.command()
def fetch():
    FetchRunner().run()


@cli.command("sync-names")
@click.option("--exchange", default=None, help="Only this exchange code (default: every venue).")
@click.option("--apply", "apply_", is_flag=True, help="Write the renames (default: only log them).")
def sync_names_command(exchange: str | None, apply_: bool):
    sync_names(init_registry(), init_db(), exchange, apply_)


@cli.command("split-securities")
@click.option("--exchange", default=None, help="Only this exchange code (default: every venue).")
@click.option("--apply", "apply_", is_flag=True, help="Write the splits (default: only log them).")
def split_securities_command(exchange: str | None, apply_: bool):
    split_securities(init_registry(), init_db(), exchange, apply_)


@cli.command("repair-overdue")
@click.option("--exchange", default=None, help="Only this exchange code (default: every venue).")
@click.option("--expired-days", default=7, show_default=True, help="Only events that expired more than this long ago.")
@click.option("--apply", "apply_", is_flag=True, help="Write settlements and resolutions (default: only log them).")
def repair_overdue_command(exchange: str | None, expired_days: int, apply_: bool):
    repair_overdue(init_registry(), init_db(), exchange, expired_days, apply_)


if __name__ == "__main__":
    cli()
