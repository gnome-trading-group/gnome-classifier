import click

from classifier.utils import setup_logging
from classifier.workers.config import WorkerConfig, init_db, init_registry
from classifier.workers.fetch import FetchRunner
from classifier.workers.normalize import NormalizeWorker
from classifier.workers.notify import NotifyWorker
from classifier.workers.relationships import RelationshipsWorker
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


if __name__ == "__main__":
    cli()
