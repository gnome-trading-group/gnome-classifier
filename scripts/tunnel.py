"""
Start SSM port-forwarding tunnels to RDS (5432) and Redis (6379) through the bastion host.

Prerequisites:
  brew install --cask session-manager-plugin
  aws sso login  (or equivalent credential setup)

Usage:
  poetry run tunnel              # both tunnels, dev stage
  poetry run tunnel --redis      # Redis only
  poetry run tunnel --pg         # Postgres only
  poetry run tunnel --stage prod # prod stage
"""
import json
import signal
import subprocess
import sys
import time

import boto3
import click


def _cfn_output(client, stack_name: str, key: str) -> str:
    stacks = client.describe_stacks(StackName=stack_name)["Stacks"]
    if not stacks:
        raise click.ClickException(f"Stack not found: {stack_name}")
    outputs = {o["OutputKey"]: o["OutputValue"] for o in stacks[0].get("Outputs", [])}
    if key not in outputs:
        raise click.ClickException(f"Output '{key}' not found in stack {stack_name}")
    return outputs[key]


def _db_secret(client, secret_name: str) -> dict:
    raw = client.get_secret_value(SecretId=secret_name)["SecretString"]
    return json.loads(raw)


def _start_tunnel(bastion_id: str, remote_host: str, remote_port: int, local_port: int) -> subprocess.Popen:
    params = json.dumps({
        "host": [remote_host],
        "portNumber": [str(remote_port)],
        "localPortNumber": [str(local_port)],
    })
    return subprocess.Popen(
        [
            "aws", "ssm", "start-session",
            "--target", bastion_id,
            "--document-name", "AWS-StartPortForwardingSessionToRemoteHost",
            "--parameters", params,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


@click.command()
@click.option("--stage", default="dev", show_default=True, help="dev or prod")
@click.option("--pg", "pg_only", is_flag=True, help="Postgres tunnel only")
@click.option("--redis", "redis_only", is_flag=True, help="Redis tunnel only")
@click.option("--pg-port", default=5432, show_default=True, help="Local port for Postgres")
@click.option("--redis-port", default=6379, show_default=True, help="Local port for Redis")
def main(stage: str, pg_only: bool, redis_only: bool, pg_port: int, redis_port: int) -> None:
    stage_title = stage.title()  # "dev" → "Dev"
    db_stack = f"{stage_title}-DatabaseStack"
    classifier_stack = f"{stage_title}-ClassifierStack"

    cfn = boto3.client("cloudformation")
    sm = boto3.client("secretsmanager")

    click.echo(f"Discovering endpoints for stage={stage}...")

    bastion_id = _cfn_output(cfn, db_stack, "BastionInstanceId")
    db = _db_secret(sm, "registry-database-root-user")
    db_host = db["host"]
    db_user = db["username"]
    db_pass = db["password"]
    db_name = db.get("dbname", "gnome")

    procs: list[subprocess.Popen] = []

    if not redis_only:
        click.echo(f"  Starting Postgres tunnel: localhost:{pg_port} → {db_host}:5432")
        procs.append(_start_tunnel(bastion_id, db_host, 5432, pg_port))

    if not pg_only:
        redis_endpoint = _cfn_output(cfn, classifier_stack, "RedisEndpoint")
        # endpoint is redis://host:port — extract host and port
        redis_parts = redis_endpoint.replace("redis://", "").split(":")
        redis_host = redis_parts[0]
        redis_port_remote = int(redis_parts[1]) if len(redis_parts) > 1 else 6379
        click.echo(f"  Starting Redis tunnel:    localhost:{redis_port} → {redis_host}:{redis_port_remote}")
        procs.append(_start_tunnel(bastion_id, redis_host, redis_port_remote, redis_port))

    # Brief pause so SSM sessions can initialise
    time.sleep(2)

    for proc in procs:
        if proc.poll() is not None:
            raise click.ClickException("An SSM session failed to start. Is the session-manager-plugin installed?")

    click.echo("\nTunnels ready. Export these env vars:\n")
    if not redis_only:
        click.echo(f"  export DATABASE_URL=postgresql://{db_user}:{db_pass}@localhost:{pg_port}/{db_name}")
    if not pg_only:
        click.echo(f"  export REDIS_URL=redis://localhost:{redis_port}")
    click.echo("\nPress Ctrl-C to close tunnels.")

    def _shutdown(signum, frame):
        click.echo("\nClosing tunnels...")
        for proc in procs:
            proc.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    for proc in procs:
        proc.wait()
