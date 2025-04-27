from lightning_sdk import Machine, Studio
from litdata import __version__

# Start the studio
s = Studio()
print("starting Studio...")
s.start()

# Install plugin if not installed (in this case, it is already installed)
s.install_plugin("jobs")

jobs_plugin = s.installed_plugins["jobs"]

# --------------------------
# Launch job to optimize the data
# --------------------------
job_cmd = f"cd litdata-benchmark && python optimize_imagenet.py --input_dir /teamspace/s3_connections/imagenet-1m-template/raw/train --output_dir /teamspace/datasets/imagenet-1m-optimized-{__version__}/train"
jobs_plugin.run(
    job_cmd,
    name=f"prepare-imagenet-1m-using-litdata-{__version__}",
    machine=Machine.DATA_PREP,
    interruptible=False,
)
