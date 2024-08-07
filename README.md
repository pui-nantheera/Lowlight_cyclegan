# Contextual Colorization and Denoising for Low-Light Ultra High Resolution Sequences

[Paper](https://ieeexplore.ieee.org/document/9506694): N. Anantrasirichai and D. Bull, "Contextual Colorization and Denoising for Low-Light Ultra High Resolution Sequences," 2021 IEEE International Conference on Image Processing (ICIP), 2021, pp. 1614-1618. [[arXiv](https://arxiv.org/pdf/2101.01597.pdf)]

---
This code was modified from CycleGAN and pix2pix in PyTorch. For installation, please follow the instructions at https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

## Data preparation

Please see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix for data structure for training.

To generate training and testing patches, please see https://github.com/pui-nantheera/Lowlight_cyclegan/tree/main/matlab_code

## Train

```
DATA_DIR="/work/datasets"
TESTDATA_DIR="/work/datasets/testA"
CHECKPOINT_DIR="/work/checkpoint"
RESULT_DIR="/work/results"

MODEL_NAME="CycleGAN_conreg_crop360_lsgan1half"
python train.py --dataroot="$DATA_DIR" --name="$MODEL_NAME" --model=cycle_gan --dataset_mode=concatregs --checkpoints_dir="$CHECKPOINT_DIR" --crop_size 360 --load_size 512 --input_nc=6 --output_nc=6  --gan_mode=lsgan1half 
```

## Test
```
python test.py --dataroot="$TESTDATA_DIR" --name="$MODEL_NAME" --model=test --dataset_mode=singleconreg --epoch 200 --checkpoints_dir="$CHECKPOINT_DIR" --results_dir="$RESULT_DIR" --no_dropout --model_suffix _A  --crop_size 512  --load_size 512 --num_test 1000  --input_nc=6 --output_nc=6
```

## Citation
```
@INPROCEEDINGS{9506694,
  author={Anantrasirichai, N. and Bull, David},
  booktitle={IEEE International Conference on Image Processing (ICIP)}, 
  title={Contextual Colorization and Denoising for Low-Light Ultra High Resolution Sequences}, 
  year={2021},
  pages={1614-1618},
  doi={10.1109/ICIP42928.2021.9506694}}
```
