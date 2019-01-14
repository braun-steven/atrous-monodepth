import matplotlib.pyplot as plt
import argparse

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--title", type=str)
parser.add_argument("--xlabel", type=str)
parser.add_argument("--ylabel", type=str)
parser.add_argument("--outpath", type=str)
parser.add_argument(
    "--xs",
    nargs="+",
    type=float,
    help="x values"
)
parser.add_argument(
    "--ys",
    nargs="+",
    type=float,
    help="y values"
)
args = parser.parse_args()

# xs = [1, 6, 12, 18, 24, 30, 36, 42]
# ys = [0.1059, 0.1049, 0.1045, 0.1061, 0.1061, 0.1068, 0.1074, 0.1054]

# xs = [1, 6, 12, 18, 24, 30, 36, 42, 48, 54]
# ys = [0.1035, 0.1058, 0.1074, 0.1075, 0.1075, 0.1073, 0.1077, 0.1075, 0.1086, 0.1074]
# title = "Influence of ASPP Modules"
# xlabel = "ASPP Max Module Size"
# ylabel = "Test set abs_rel"
# outpath = "aspp.png"

xs = args.xs
ys = args.ys
title = args.title
xlabel = args.xlabel
ylabel = args.ylabel
outpath = args.outpath

plt.figure()
plt.title(title)
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.ylim((0.0975, 0.115))
plt.scatter(xs, ys, marker="x")
plt.savefig(outpath)


