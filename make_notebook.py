"""
make_notebook.py
Builds submissions/kaggle_notebook.ipynb from the current submit.py.
The notebook contains:
  Cell 0: Markdown header
  Cell 1: %%writefile submit.py  (full solver inlined)
  Cell 2: Environment probe (cpu count, input files)
  Cell 3: Run solver via subprocess
  Cell 4: Preview submission CSV

Run:  python make_notebook.py
Output: submissions/kaggle_notebook.ipynb
"""
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
SUBMIT_PY = os.path.join(ROOT, "submit.py")
OUT_NB    = os.path.join(ROOT, "submissions", "kaggle_notebook.ipynb")

with open(SUBMIT_PY, encoding="utf-8") as f:
    solver_lines = f.readlines()


def code_cell(source_lines):
    """Create a Kaggle-compatible code cell from a list of source lines."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source_lines,
    }


def md_cell(text):
    return {
        "cell_type": "markdown",
        "id": "md",
        "metadata": {},
        "source": [text],
    }


# ---------------------------------------------------------------------------
# Cell 1 — writefile magic + full solver source
# Cell source must be a list of strings (each line ending with \n).
# The %%writefile magic must be the FIRST line.
# ---------------------------------------------------------------------------
writefile_cell_lines = ["%%writefile submit.py\n"] + solver_lines

# ---------------------------------------------------------------------------
# Cell 2 — env probe
# ---------------------------------------------------------------------------
env_probe = code_cell([
    "import os\n",
    "print('CPUs available:', os.cpu_count())\n",
    "print('Input files:', os.listdir('/kaggle/input/santa-2026/'))\n",
    "print('submit.py written:', os.path.exists('submit.py'))\n",
])

# ---------------------------------------------------------------------------
# Cell 3 — run solver
# We use --cores (cpu_count - 1), --sa-time 30 (Kaggle allows ~9 hrs),
# --n-starts 3, and --verify so we get a score printout.
# ---------------------------------------------------------------------------
run_cell_src = [
    "%%time\n",
    "import subprocess, sys, os\n",
    "\n",
    "PUZZLES  = '/kaggle/input/santa-2026/puzzles.csv'\n",
    "OUT      = '/kaggle/working/submission.csv'\n",
    "CORES    = max(1, (os.cpu_count() or 2) - 1)   # leave 1 CPU free\n",
    "SA_TIME  = 30.0   # seconds SA budget per puzzle (increase for better score)\n",
    "N_STARTS = 3      # MiM random restarts\n",
    "\n",
    "print(f'Cores: {CORES}, SA budget: {SA_TIME}s/puzzle')\n",
    "\n",
    "result = subprocess.run([\n",
    "    sys.executable, 'submit.py',\n",
    "    '--puzzles',  PUZZLES,\n",
    "    '--out',      OUT,\n",
    "    '--cores',    str(CORES),\n",
    "    '--sa-time',  str(SA_TIME),\n",
    "    '--n-starts', str(N_STARTS),\n",
    "    '--verify',\n",
    "], capture_output=False, text=True)\n",
    "\n",
    "print('Return code:', result.returncode)\n",
    "assert result.returncode == 0, 'Solver failed!'\n",
]
run_cell = code_cell(run_cell_src)

# ---------------------------------------------------------------------------
# Cell 4 — sanity preview
# ---------------------------------------------------------------------------
preview_cell = code_cell([
    "import pandas as pd\n",
    "sub = pd.read_csv('/kaggle/working/submission.csv')\n",
    "print('Rows:', len(sub))\n",
    "print('Columns:', list(sub.columns))\n",
    "print(sub.head(3))\n",
    "# Count total instructions\n",
    "total = sub['program'].apply(lambda x: len(__import__('json').loads(x))).sum()\n",
    "print(f'Total instructions: {total:,}')\n",
])

# ---------------------------------------------------------------------------
# Assemble notebook
# ---------------------------------------------------------------------------
nb = {
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
    "cells": [
        md_cell(
            "# Santa 2026 — The Elf Calculator\n\n"
            "**Strategy:** Pool-aware binary construction (shared-prefix) + simulated annealing post-pass.\n\n"
            "**How to use:**\n"
            "1. Run **Cell 1** (writes `submit.py` to the working directory)\n"
            "2. Run **Cell 2** (env probe — confirm CPUs and input files)\n"
            "3. Run **Cell 3** (solves all 1,000 puzzles → `submission.csv`)\n"
            "4. Run **Cell 4** (sanity check)\n"
            "5. Submit `submission.csv` via *Submit* button.\n\n"
            "> **SA budget:** `SA_TIME = 30.0` s/puzzle. Kaggle allows up to ~9 hrs total. "
            "Increase to 60+ for better scores."
        ),
        code_cell(writefile_cell_lines),
        env_probe,
        run_cell,
        preview_cell,
    ],
}

os.makedirs(os.path.dirname(OUT_NB), exist_ok=True)
with open(OUT_NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

nb_size_kb = os.path.getsize(OUT_NB) / 1024
print(f"Written: {OUT_NB}  ({nb_size_kb:.1f} KB)")
print(f"  Solver source lines embedded: {len(solver_lines)}")
print(f"  Total notebook cells: {len(nb['cells'])}")
