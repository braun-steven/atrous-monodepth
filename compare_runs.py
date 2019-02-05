from PIL import Image, ImageDraw, ImageFont
import os
import csv
import argparse

parser = argparse.ArgumentParser(
    description="Resize and concatenate output images of multiple runs for experimental comparison"
)

parser.add_argument("--rundir", type=str, help="run directory")
parser.add_argument("--runs", type=str, help="run folders")
parser.add_argument("--output", type=str, help="put the concatenated images there")

parser.add_argument(
    "--kittidir",
    type=str,
    default="data/kitti/",
    help="directory where kitti test data lies",
)
parser.add_argument(
    "--kitti-extension",
    type=str,
    default="png",
    help="file extension for kitti files (jpg or png)",
)

args = parser.parse_args()
runs = args.runs.split(",")
num_images = len(runs) + 1

resolution = (1242, 375)

# Setup folders
rootdir = args.rundir
testdirs = [os.path.join(run, "test", "preds") for run in runs]
imagedir = os.path.join(args.kittidir, "training/image_2")
outputdir = args.output

if not os.path.exists(outputdir):
    os.mkdir(outputdir)

width = 1242
total_height = 375 * num_images

try:
    fnt = ImageFont.truetype("/Library/Fonts/Arial.ttf", 15)
except OSError:
    fnt = ImageFont.truetype("/usr/share/fonts/TTF/Arial.ttf", 15)

for i in range(200):
    img = Image.open(
        os.path.join(imagedir, "{}_10.{}".format(str(i).zfill(6), args.kitti_extension))
    )
    new_im = Image.new("RGB", (width, total_height))
    new_im.paste(img, (0, 0))

    for j, run in enumerate(runs):
        testdir = os.path.join(rootdir, runs[j], "test", "preds")
        infile = "{}.png".format(str(i).zfill(3))
        infile = os.path.join(testdir, infile)

        disp = Image.open(infile)
        disp = disp.resize(resolution, Image.ANTIALIAS)

        d = ImageDraw.Draw(disp)
        d.text((10, 350), runs[j], font=fnt, fill=(255, 255, 255))

        new_im.paste(disp, (0, (j + 1) * 375))

    concat_outfile = "concat_{}.png".format(str(i).zfill(3))
    concat_outfile = os.path.join(outputdir, concat_outfile)
    new_im.save(concat_outfile)

for run in runs:
    print(run)
    scores = os.path.join(rootdir, run, "test", "scores.csv")
    with open(scores, newline="") as File:
        reader = csv.reader(File)
        for row in reader:
            print(row)
