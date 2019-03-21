import os

folder = "results/milestones"
for directory in [f.path for f in os.scandir(folder) if f.is_dir()]:
    for rundir in [f.path for f in os.scandir(directory) if f.is_dir()]:
        print(rundir)
