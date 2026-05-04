"""Command line interface for llmscope."""

from __future__ import annotations

import typer

from . import __version__

app = typer.Typer(help="llmscope command line interface")


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Start the llmscope API server."""
    raise typer.Exit()


@app.command()
def instrument() -> None:
    """Prepare instrumentation for model tracing."""
    raise typer.Exit()


@app.command()
def version() -> None:
    """Print llmscope version."""
    typer.echo(__version__)


def main() -> None:
    app()
