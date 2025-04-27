import os
from time import time
from lightning_sdk import Studio, Machine
from datetime import datetime


# Create a new Studio
studio_name = f"imagenet-benchmark-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
studio = Studio(name=studio_name)
studio.start(machine=Machine.A10G, interruptible=False)

try:
    print(" ")
    print(f"A10G Studio created: {studio_name}")
    print(" ")
    print("######################")
    print(" ")

    print("Setting up...")
    print(" ")

    # Setup: Upload all python files + install dependencies
    for folder, _, filenames in os.walk("."):
        for filename in filenames:
            if not filename.endswith(".py"):
                continue
            path = os.path.join(folder, filename)
            studio.upload_file(path, path.replace("./", ""))

    studio.upload_file("requirements.txt", "requirements.txt")
    studio.run("pip install -U -r requirements.txt")

    studio.run("sudo apt-get install file")
    studio.run("conda config --add channels conda-forge")
    studio.run("conda config --set channel_priority strict")
    studio.run("conda install s5cmd > /dev/null 2>&1")

    print()
    print("######################")
    print()

    # Benchmark litdata
    t0 = time()
    print("LitData Data Logs:")
    print()
    print(studio.run("python stream_imagenet.py"))
    print()
    print(f"LitData Data  executed in {time() - t0}")

    print()
    print("######################")
    print()

    print("Stopping Studio...")
    studio.stop()
    print("Done.")

except Exception as e:
    print(e)

studio.delete()
