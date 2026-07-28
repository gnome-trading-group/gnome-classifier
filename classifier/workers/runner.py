import logging

import click

from classifier.workers.config import WorkerConfig
from classifier.workers.embed import EmbedWorker
from classifier.workers.normalize import NormalizeWorker
from classifier.workers.notify import NotifyWorker
from classifier.workers.relationships import RelationshipsWorker


@click.group()
def cli():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


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


if __name__ == "__main__":
    cli()
