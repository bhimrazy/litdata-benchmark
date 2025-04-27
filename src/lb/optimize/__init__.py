import typer
from lb.optimize.imagenet import optimize_imagenet

optimize_app = typer.Typer(help="Optimize datasets for benchmarking.")
optimize_app.command("imagenet")(optimize_imagenet)
