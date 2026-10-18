"""Execute the exploration from a fresh kernel using this Python environment."""

import json
import os
from pathlib import Path
import sys
import tempfile

import nbformat
from nbclient import NotebookClient


def main():
    root = Path(__file__).resolve().parents[1]
    path = root / "notebooks" / "01_data_exploration.ipynb"
    notebook = nbformat.read(path, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    # A temporary kernel specification avoids installing a global/user kernel.
    previous = os.environ.get("JUPYTER_PATH")
    try:
        with tempfile.TemporaryDirectory() as directory:
            kernel = Path(directory) / "kernels" / "churn-exploration"
            kernel.mkdir(parents=True)
            (kernel / "kernel.json").write_text(json.dumps({
                "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                "display_name": "Churn exploration", "language": "python",
            }), encoding="utf-8")
            os.environ["JUPYTER_PATH"] = os.pathsep.join(
                part for part in [directory, previous] if part)
            NotebookClient(notebook, timeout=180, kernel_name="churn-exploration",
                           resources={"metadata": {"path": str(root)}}).execute()
    finally:
        if previous is None:
            os.environ.pop("JUPYTER_PATH", None)
        else:
            os.environ["JUPYTER_PATH"] = previous
    for cell in notebook.cells:
        cell.metadata.pop("execution", None)
    nbformat.validate(notebook)
    nbformat.write(notebook, path)
    print("Fresh-kernel exploration completed; notebook and aggregate outputs saved.")


if __name__ == "__main__":
    main()
