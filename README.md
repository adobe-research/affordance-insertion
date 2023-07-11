# Putting People in Their Place: Affordance-Aware Human Insertion into Scenes
### [Project Page](https://sumith1896.github.io/affordance-insertion/) | [Paper](https://sumith1896.github.io/affordance-insertion/static/paper/affordance_insertion_cvpr2023.pdf)
This repository contains the original PyTorch implementation of this project. <br>

[Putting People in Their Place: Affordance-Aware Human Insertion into Scenes](https://sumith1896.github.io/affordance-insertion/)<br> 
 [Sumith Kulal](https://cs.stanford.edu/~sumith/)<sup>1</sup>,
 [Tim Brooks](https://timothybrooks.com/about)<sup>2</sup>,
 [Alex Aiken](http://theory.stanford.edu/~aiken/)<sup>1</sup>,
 [Jiajun Wu](https://jiajunwu.com/)<sup>1</sup>, <br>
 [Jimei Yang](https://jimeiyang.github.io/)<sup>3</sup>,
 [Jingwan Lu](https://research.adobe.com/person/jingwan-lu/)<sup>3</sup>,
 [Alexei A. Efros](https://people.eecs.berkeley.edu/~efros/)<sup>2</sup>,
 [Krishna Kumar Singh](http://krsingh.cs.ucdavis.edu/)<sup>3</sup> <br>
 <sup>1</sup>Stanford University, <sup>2</sup>UC Berkeley, <sup>3</sup>Adobe Research <br>
  
  <img src='https://sumith1896.github.io/affordance-insertion/static/images/teaser.png'/>


## Setup

Install the necessary packages using either `pip` or `conda`:

```
# install via pip (recommended)
python3 -m venv .affordance
source .affordance/bin/activate
pip install wheel
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 \
	torchdata==0.3.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt

# install via conda
conda env create -f environment.yaml
```
  
## Dataset

Similar to [Hallucinating Pose-Compatible Scenes](https://github.com/timothybrooks/hallucinating-scenes), we preprocess raw input video data into LMDBs that can be ingested by our dataloaders and training pipeline. Starting from raw input video data, we first process it to only consist spatio-temporal sequences of single human motion. We then pre-compute the human segmentation masks. A sample sliver of data in LMDB format is [here](https://huggingface.co/datasets/sumith1896/affordance_sample_data).

Our pre-processing scripts also closely follow [hallucinating-scenes](https://github.com/timothybrooks/hallucinating-scenes) repo, with improvements for speed and mask generation.

### Preprocessing steps  

You can refer to more complete instructions [here](https://github.com/timothybrooks/hallucinating-scenes/blob/master/dataset.md).

```
# preprocess vidoes, use images script if videos are in extracted format
python data_from_videos.py input_dir=kinetics output_dir=kinetics_frames_256
### python data_from_images.py input_dir=kinetics output_dir=kinetics_256

# filter by detecting people bbox
python data_filter_people.py input_dir=kinetics_256 output_dir=kinetics_people

# filter by detecting body keypoints
mkdir open_pose/pretrained/
gdown --id 1k7Teg2bVxGBR7ECiNScNzlLF0PDiIgmo -O open_pose/pretrained/open_pose.pt
python data_detect_pose.py input_dir=kinetics_people output_dir=kinetics_pose

# generate masks for filtered data
python data_generate_masks.py input_dir=kinetics_pose output_dir=kinetics_mask
```

Save the dataset in the following format:
```
data_root/
  --> dataset1/
    --> frames_db/
    --> masks_db/
    --> clipmask_db/
  --> dataset2/
...

```

## Training

The training follows config-driven approach as in the original [CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion) repo. 

### Dataloader

The main control for the data-loading is through the `data` config. The various config parameters have been described in detail in the `./ldm/data/hic.py` file. One could visualize the various options and the data sampled from them through them by running `python -m ldm.data.hic`. As mentioned previously, a sample dataset with handful of videos is made available [here](https://huggingface.co/datasets/sumith1896/affordance_sample_data) solely for data exploration and visualization. These videos originate from a single batch MPII Human Pose dataset and we urge users to go through the author's license [here](http://human-pose.mpi-inf.mpg.de/#download). 

### Training

Run the script `bash train.sh` for single node training. To experiment with different conditioning modalities and types (concat, crossattn), you can set the `cond_stage_key` param which has been expanded to handle more general structures.

## FAQs

- **At times there is excessive memory usage and EOM during training?** Although this rarely happens, if you observe unusual memory consumption it could be due to LMDB loading. We observed using fewer larger LMDBs help alleviate such issues. You can use merge LMDBs for smaller datasets using `ldm/data/data_merge_lmdb.py`.

- **Can you share the full dataset used for training?** We will not be able to share the datasets for various reasons including size and source copyright. Our data preparation scripts can construct input datasets from various video sources. A good starting point would large video datasets that are publicly available, as listed [here](https://github.com/timothybrooks/hallucinating-scenes/blob/master/dataset.md). 

Please reach out to Sumith (sumith@cs.stanford.edu) for any questions or concerns.

## Citation

If you find our project useful, please cite the following:
  ```
@inproceedings{kulal2023affordance,
author    = {Kulal, Sumith and Brooks, Tim and Aiken, Alex and Wu, Jiajun and Yang, Jimei and Lu, Jingwan and Efros, Alexei A. and Singh, Krishna Kumar},
title     = {Putting People in Their Place: Affordance-Aware Human Insertion into Scenes},
booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
year      = {2023},
}
  ```
