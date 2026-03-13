# Geo-SelfSSC

# 🏗️️ Setup

### Python Environment

We use **Conda** to manage our Python environment:
```shell
conda env create -f environment.yml
```
Then, activate the conda environment :
```shell
conda activate GeoSelfSSC
```

### 💾 Datasets

All non-standard data (like precomputed poses and datasplits) comes with this repository and can be found in the `datasets/` folder.
In addition, please adjust the `data_path` ,`data_depth_path`, `data_depth_std_path`, `data_normal_path` and `data_segmentation_path` in `configs/data/kitti_360.yaml`.\
We explain how to obtain these datasets in [KITTI-360](#KITTI-360) ,[Geometric Cues](#Geometric-Cues)
and [Pseudo-Ground-Truth Segmentation masks](#pseudo-ground-truth-segmentation-masks).


### KITTI-360

To download KITTI-360, go to https://www.cvlibs.net/datasets/kitti-360/index.php and create an account.
We require the perspective images, fisheye images, raw velodyne scans, calibrations, and vehicle poses.

### Geometric Cues
We use the existing SOTA depth estimators ([supervised](https://github.com/hisfog/SfMNeXt-Impl) and [self-supervised](https://github.com/nianticlabs/monodepth2)) to predict the depth cues. 
After the depth map is obtained, the normal result is further predicted.


### Pseudo-Ground-Truth Segmentation masks

We use the [Panoptic Deeplab model zoo (CVPR 2020)](https://github.com/bowenc0221/panoptic-deeplab/tree/master).
First create and activate a new conda environment following the instructions laid out [here](https://github.com/bowenc0221/panoptic-deeplab/blob/master/tools/docs/INSTALL.md). \
You can find the `requirements.txt` file under `\datasets\panoptic-deeplab\requirements.txt`.
You also need to download the [R101-os32 cityscapes baseline model](https://github.com/bowenc0221/panoptic-deeplab/blob/master/tools/docs/MODEL_ZOO.md).

Afterwards, you can run:

```bash
python <path-to-script>/preprocess_kitti_360_segmentation.py \
--cfg datasets/panoptic-deeplab/configs/panoptic_deeplab_R101_os32_cityscapes.yaml \
--output-dir <path-to-output-directory> \
--checkpoint <path-to-downloaded-model>/panoptic_deeplab_R101_os32_cityscapes.pth
```

# 🏋 Training

The training configuration for the model reported on in the paper can be found in the `configs` folder.
Generally, all trainings are run on 2 Nvidia RTX4090 Gpus with a total memory of 48GB. 
For faster convergence and slightly better results, we use the pretrained model from [BehindTheScenes](https://fwmb.github.io/bts/)
as a backbone from which we start our training. To download the backbone please run:

```bash
./download_backbone.sh
```

**KITTI-360**

```bash
python train.py -cn exp_kitti_360
```

# 📊 Evaluation

We provide **not only** a way to evaluate our method (Geo-SelfSSC) on the SSCBench KITTI-360 dataset, 
but also a way to easily evaluate/compare other methods. For this, you only need the predictions on the test set 
(sequence 09) saved as `frame_id.npy` files in a folder. \

## Geo-SelfSSC on SSCBench KITTI-360

To evaluate our model on the SSCBench KITTI-360 dataset, we need additional data:

### SSCBench KITTI-360 dataset

We require the SSCBench KITTI-360 dataset, which can be downloaded from [here](https://github.com/ai4ce/SSCBench/tree/main/dataset/KITTI-360).

### SSCBench KITTI-360 ground truth

We also need preprocessed ground truth (voxelized ground truth) that belongs to the KITTI-360 SSCBench data. 
The preprocessed data for KITTI-360 in the GitHub Repo was incorrectly generated ([see here](https://github.com/ai4ce/SSCBench/issues/9)).\
Therefore, we provide the pre-processed ground truth verified in s4c for download [here](https://cvg.cit.tum.de/webshare/g/s4c/voxel_gt.zip).

You can now run the evaluation script found at `scripts/benchmarks/sscbench/evaluate_model_sscbench.py` by running:

```bash
python evaluate_model_sscbench.py \
-ssc <path-to-kitti_360-sscbench-dataset> \
-vgt <path-to-preprocessed-voxel-ground-truth> \
-cp <path-to-model-checkpoint> \
-f
```


# 🗣️ Acknowledgements

This repository is based on the [S4C](https://github.com/ahayler/s4c) and [BehindTheScenes](https://fwmb.github.io/bts/). 
We evaluate our models on the novel [SSCBench KITTI-360 benchmark](https://github.com/ai4ce/SSCBench). 
We generate our pseudo 2D ground truth using the [Monodepth2](https://github.com/nianticlabs/monodepth2), [SQLdepth](https://github.com/hisfog/SfMNeXt-Impl) and [Panoptic Deeplab model zoo](https://github.com/bowenc0221/panoptic-deeplab/tree/master).
