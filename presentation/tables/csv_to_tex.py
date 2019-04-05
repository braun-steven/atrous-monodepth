import sys
import pandas as pd
in_path = sys.argv[1]
df = pd.read_csv(in_path)
out_path = in_path.split(".")[0] + ".tex"
with open(out_path, "w") as f:
    df.to_latex(f)
