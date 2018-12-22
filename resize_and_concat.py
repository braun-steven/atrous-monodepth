from PIL import Image
import os
import argparse
from subprocess import Popen, PIPE

parser = argparse.ArgumentParser(
    description="Resize and concatenate output images of a run."
)
parser.add_argument("--rundir", type=str, help="root directory of the run")
parser.add_argument(
    "--kittidir",
    type=str,
    default="data/kitti/",
    help="directory where kitti test data lies",
)

args = parser.parse_args()

resolution = (1242, 375)

rootdir = args.rundir
testdir = os.path.join(rootdir, "test", "preds")

imagedir = os.path.join(args.kittidir, "training/image_2")


if not os.path.exists(os.path.join(testdir, "concat")):
    os.mkdir(os.path.join(testdir, "concat"))

if not os.path.exists(os.path.join(testdir, "resized")):
    os.mkdir(os.path.join(testdir, "resized"))


width = 1242
total_height = 375 * 2
for i in range(200):

    infile = "{}.png".format(str(i).zfill(3))
    infile = os.path.join(testdir, infile)

    img = Image.open(os.path.join(imagedir, "{}_10.jpg".format(str(i).zfill(6))))

    disp = Image.open(infile)
    disp = disp.resize(resolution, Image.ANTIALIAS)

    outfile = "pred_{}_resized.png".format(str(i).zfill(3))
    outfile = os.path.join(testdir, "resized", outfile)
    disp.save(outfile)

    new_im = Image.new("RGB", (width, total_height))
    new_im.paste(img, (0, 0))
    new_im.paste(disp, (0, 375))

    concat_outfile = "concat_{}.png".format(str(i).zfill(3))
    concat_outfile = os.path.join(testdir, "concat", concat_outfile)
    new_im.save(concat_outfile)


# Create gif output path
gif_out_path = os.path.join(testdir, "test.mp4")

# Command using ffmpeg
cmd = f"ffmpeg -f image2 -framerate 2 -i {os.path.join(testdir, 'concat')}/concat_%003d.png {gif_out_path}"

# Run command
process = Popen(cmd.split(), stdout=PIPE, stderr=PIPE)
process.communicate()
