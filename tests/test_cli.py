import subprocess

def test_cli_help():
    result = subprocess.run([
        "python", "-m", "lb.cli", "--help"
    ], capture_output=True, text=True)
    assert result.returncode == 0
    assert "LitData Benchmarking CLI" in result.stdout

def test_optimize_imagenet_help():
    result = subprocess.run([
        "python", "-m", "lb.cli", "optimize", "imagenet", "--help"
    ], capture_output=True, text=True)
    assert result.returncode == 0
    assert "Optimize ImageNet dataset for benchmarking" in result.stdout
