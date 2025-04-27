import typer
from lb.stream.imagenet import stream_imagenet

stream_app = typer.Typer(help="Benchmark streaming datasets.")
stream_app.command("imagenet")(stream_imagenet)
