import os

content = """No dataset found at {}.
    
Raw Imagenet dataset doesn't exist or is empty.
Please make sure raw imagenet dataset is available for optimizing and benchmarking.
"""

def is_non_empty_directory(path: str) -> bool:
    """Check if a directory exists and is non-empty."""
    return os.path.isdir(path) and bool(os.listdir(path))

def write_to_file(path: str, content: str) -> None:
    """Write content to a file."""
    with open(path, 'w') as f:
        f.write(content)

if __name__ == "__main__":
    dir_path = "/teamspace/s3_connections/imagenet-1m-template/raw/train"

    does_exists = is_non_empty_directory(dir_path)

    if does_exists:
        write_to_file("result.md", content.format(dir_path))
