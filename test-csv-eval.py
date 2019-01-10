import glob
import os
import sys
import pandas as pd
import argparse
import matplotlib.pyplot as plt

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--result-dir", default="results", type=str)
parser.add_argument(
    "--runs",
    nargs="+",
    type=str,
    help="Run regex",
)
args = parser.parse_args()

result_dir = args.result_dir

# Find all files specified by the glob list
files = []
for run in args.runs:
    gl = os.path.join(result_dir, run, "test", "scores.csv")
    run_files = glob.glob(gl)
    files += run_files

# Load scores.csv as dataframes
dfs = []
for score_file in files:
    df = pd.read_csv(score_file, header=0, delimiter=",")
    df.drop(df.index[1], inplace=True)
    dfs.append(df)
    
# Postprocess and merge dataframes
df = pd.concat(dfs)
files = [f.split(":", maxsplit=1)[1][4:].split("/test/")[0] for f in files]
df["run"] = pd.Series(files, index=df.index)
df.columns = [s.strip() for s in df.columns]
df = df[df["abs_rel"] < 5.0]

# Print stats and overview 
print(df.sort_values(by="abs_rel"))
print(df.describe())

