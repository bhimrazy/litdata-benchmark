import typer

from lb.optimize import optimize_app
from lb.stream import stream_app

app = typer.Typer(help="LitData Benchmarking CLI")


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    if ctx.invoked_subcommand is None:
        typer.secho(
            "\n✨ Welcome to LitData Benchmark CLI! ✨\n",
            fg=typer.colors.CYAN,
            bold=True,
        )
        typer.echo(
            "Easily benchmark and optimize your datasets with a modern, modular CLI."
        )
        typer.echo("Type 'lb --help' to see all available commands.\n")
        typer.echo(ctx.get_help())


app.add_typer(optimize_app, name="optimize")
app.add_typer(stream_app, name="stream")

if __name__ == "__main__":
    app()
