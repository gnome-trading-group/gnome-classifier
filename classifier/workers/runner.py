import click

from classifier.utils import setup_logging
from classifier.workers.config import WorkerConfig
from classifier.workers.embed import EmbedWorker
from classifier.workers.fetch import FetchRunner
from classifier.workers.normalize import NormalizeWorker
from classifier.workers.notify import NotifyWorker
from classifier.workers.relationships import RelationshipsWorker


@click.group()
def cli():
    setup_logging()


@cli.command()
def normalize():
    NormalizeWorker(WorkerConfig()).run()


@cli.command()
def embed():
    EmbedWorker(WorkerConfig()).run()


@cli.command()
def relationships():
    RelationshipsWorker(WorkerConfig()).run()


@cli.command()
def notify():
    NotifyWorker(WorkerConfig()).run()


@cli.command()
def fetch():
    FetchRunner().run()


if __name__ == "__main__":
    cli()
