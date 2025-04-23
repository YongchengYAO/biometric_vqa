import math
import argparse
from pathlib import Path


# =========================
# Usage:
# Split a large file into chunks:
#   python large_file_handler.py split /path/to/large/file.nii.gz --size 40000
# Join chunks back into the original file:
#   python large_file_handler.py join /path/to/file.nii.gz.chunks
# =========================


def split_file(file_path, chunk_size_mb=1000):
    """
    Split a large file into smaller chunks

    Args:
        file_path: Path to the file to split
        chunk_size_mb: Size of each chunk in megabytes (default 1000MB = ~1GB)
    """
    file_path = Path(file_path)
    if not file_path.exists():
        print(f"Error: File {file_path} not found!")
        return

    # Convert MB to bytes
    chunk_size = chunk_size_mb * 1024 * 1024
    file_size = file_path.stat().st_size

    # Calculate number of chunks needed
    num_chunks = math.ceil(file_size / chunk_size)

    print(
        f"Splitting {file_path} ({file_size/1024/1024/1024:.2f} GB) into {num_chunks} chunks of {chunk_size_mb} MB each"
    )

    # Create a directory for the chunks
    chunk_dir = file_path.with_suffix(".chunks")
    chunk_dir.mkdir(exist_ok=True)

    # Create a manifest file
    with open(chunk_dir / "manifest.txt", "w") as manifest:
        manifest.write(f"original_file: {file_path.name}\n")
        manifest.write(f"total_size: {file_size}\n")
        manifest.write(f"chunk_size: {chunk_size}\n")
        manifest.write(f"num_chunks: {num_chunks}\n")

    # Split the file
    with open(file_path, "rb") as f:
        for i in range(num_chunks):
            chunk_file = chunk_dir / f"{file_path.name}.part{i:04d}"
            print(f"  Creating chunk {i+1}/{num_chunks}: {chunk_file.name}")

            with open(chunk_file, "wb") as chunk:
                chunk_data = f.read(chunk_size)
                chunk.write(chunk_data)

    print(f"Split complete! Chunks stored in {chunk_dir}")
    return chunk_dir


def join_file(chunks_dir):
    """
    Join file chunks back into the original file

    Args:
        chunks_dir: Directory containing the chunks and manifest
    """
    chunks_dir = Path(chunks_dir)
    if not chunks_dir.exists() or not chunks_dir.is_dir():
        print(f"Error: Chunks directory {chunks_dir} not found!")
        return

    manifest_path = chunks_dir / "manifest.txt"
    if not manifest_path.exists():
        print(f"Error: Manifest file not found in {chunks_dir}")
        return

    # Parse manifest
    manifest = {}
    with open(manifest_path, "r") as f:
        for line in f:
            key, value = line.strip().split(": ", 1)
            manifest[key] = value

    original_filename = manifest["original_file"]
    num_chunks = int(manifest["num_chunks"])

    # Path for the restored file (in parent directory of chunks)
    output_file = chunks_dir.parent / original_filename

    print(f"Joining {num_chunks} chunks into {output_file}")

    with open(output_file, "wb") as outfile:
        for i in range(num_chunks):
            chunk_file = chunks_dir / f"{original_filename}.part{i:04d}"
            print(f"  Adding chunk {i+1}/{num_chunks}: {chunk_file.name}")

            if not chunk_file.exists():
                print(f"Error: Chunk file {chunk_file} not found!")
                return

            with open(chunk_file, "rb") as chunk:
                outfile.write(chunk.read())

    print(f"Join complete! Restored file: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Split and join large files to work around size limitations"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Split command
    split_parser = subparsers.add_parser("split", help="Split a file into chunks")
    split_parser.add_argument("file", help="File to split")
    split_parser.add_argument(
        "--size",
        type=int,
        default=1000,
        help="Size of each chunk in MB (default: 1000)",
    )

    # Join command
    join_parser = subparsers.add_parser("join", help="Join chunks back into a file")
    join_parser.add_argument("chunks_dir", help="Directory containing chunks")

    args = parser.parse_args()

    if args.command == "split":
        split_file(args.file, args.size)
    elif args.command == "join":
        join_file(args.chunks_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
