"""
Santa 2026 — Kaggle submission notebook (run as script on Kaggle)

Upload this file + submit.py to a Kaggle notebook.
OR: paste submit.py into a single notebook cell and run it.

Kaggle data paths:
  /kaggle/input/santa-2026/puzzles.csv
  /kaggle/input/santa-2026/practice_puzzles.csv
  /kaggle/input/santa-2026/sample_submission.csv

Output:
  /kaggle/working/submission.csv

Usage on Kaggle:
  !python submit.py --puzzles /kaggle/input/santa-2026/puzzles.csv \
                    --out /kaggle/working/submission.csv \
                    --cores 4 --sa-time 5.0 --verify
"""
import os
import subprocess
import sys

PUZZLES = "/kaggle/input/santa-2026/puzzles.csv"
OUT = "/kaggle/working/submission.csv"
CORES = 4        # Kaggle P100 GPU kernels have 4 CPUs
SA_TIME = 5.0    # More time budget on Kaggle (no local PC constraint)
N_STARTS = 3

cmd = [
    sys.executable, "submit.py",
    "--puzzles", PUZZLES,
    "--out", OUT,
    "--cores", str(CORES),
    "--sa-time", str(SA_TIME),
    "--n-starts", str(N_STARTS),
    "--verify",
]

print("Running:", " ".join(cmd))
result = subprocess.run(cmd, check=True)
print(f"\nDone. Submission at: {OUT}")
print(f"File size: {os.path.getsize(OUT) / 1024:.1f} KB")
