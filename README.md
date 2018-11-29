# Monocular Depth Estimation with Atrous Convolutions
This repo contains code for the Deep Learning in Computer Vision practical course at TU Darmstadt.

## Useful Links
### Original implementations
- [DeepLab v3+](https://github.com/tensorflow/models/tree/master/research/deeplab)
- [Unsupervised Monodepth](https://github.com/mrharicot/monodepth)
### PyTorch implementations
- [DeepLab v3+](https://github.com/jfzhang95/pytorch-deeplab-xception)
- [Unsupervised Monodepth](https://github.com/ClubAI/MonoDepth-PyTorch)

## Coding Style
- Code Formatter: [black](https://github.com/ambv/black)
- Docstring: [Google Style](https://www.chromium.org/chromium-os/python-style-guidelines)

## How to run
The code runs on Python 3.6, Pytorch 0.4.1 and has a couple of other requirements, which are stated in `requirements.txt` and can be easily installed in a virtualenv by running `setup-env.sh`.

### Training
You can train a model by running `main.py`, which is parametrized with the following arguments:
- `--model`: model type, e.g. "resnet50_md" or "deeplab"
- `--filenames-file`: file that contains a list of filenames for training (left and right image separated by a space)
- `--val-filenames-file`: same as above, but for validation set
- `--data-dir`: root directory, filenames in filename files are relative to this path
- `--output-dir`: path for results generated during a run (tensorboard, plots, checkpoints)
- `--epochs`: number of epochs to be trained
- `--batch-size`: number of images per batch
- `--cuda-device-ids`: GPUs on which to train, -1 for cpu only

### Testing

### Evaluation
