# YOLO STAMP

Stamp detection via YOLO 

## Installation

First, make sure **uv** is installed: https://docs.astral.sh/uv/getting-started/installation/

```bash
git clone https://github.com/karatedava/yolo_stamp.git
cd yolo_stamp
uv sync --all-extras --dev
```

## Running the CLI

```bash
uv run run_yolo_CLI.py --help
```

This will show all available commands and options.

## Quick Example

```bash
uv run run_yolo_CLI.py --input path/to/your/image.jpg
```

Enjoy stamping with YOLO!