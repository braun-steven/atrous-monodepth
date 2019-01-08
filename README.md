# Monocular Depth Estimation with Atrous Convolutions
This repo contains code for the Deep Learning in Computer Vision practical course at TU Darmstadt. The project's goal is to test whether atrous convolutions (convolutions with dilated kernels) can improve monocular depth estimation. We base our implementation upon [Unsupervised single image depth prediction with CNNs](https://github.com/mrharicot/monodepth) and improve the ResNet backbone using ideas from the semantic segmentation network [DeepLab v3+](https://github.com/tensorflow/models/tree/master/research/deeplab). Atrous convolutions might help in using information at different spatial scales without introducing lots of new parameters.

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
Testing on a dataset of choice is automatically performed after training (if the argument `--test-filenames-file` is set). If you want to manually test a given model, running `test.py` and providing a `--checkpoint` will also work.

### Evaluation
Evaluation is implemented on the KITTI Stereo 2015 dataset or the Eigen split of the KITTI dataset. It is automatically evoked after training if `--eval` is set to `kitti-gt` or `eigen`. If you want to manually run evaluation on a given `.npy` file containing the output disparities on the respective dataset, you can do so by running `eval.py`.
