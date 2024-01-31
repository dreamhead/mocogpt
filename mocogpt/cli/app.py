import os
import sys

import click

from mocogpt.cli._runner import CliRunner


@click.group()
def app():
    pass


@click.command()
@click.argument("config", nargs=1)
@click.option("--port", "-p", default=12306, help="Port to run the server on.")
def start(config, port):
    if not os.path.exists(config):
        click.echo(f"Error: Config file {config} does not exist.")
        sys.exit(1)

    CliRunner().run(config, port)


app.add_command(start)

if __name__ == '__main__':
    app()
