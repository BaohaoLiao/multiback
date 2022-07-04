# Multilingual Backtranslation
This repository provides scripts, including preprocessing and training, for our WMT21's paper, [Back-translation for Large-Scale Multilingual Machine Translation](https://aclanthology.org/2021.wmt-1.50/).

## Key Points of Our Paper
* We achieved the 2nd space for small tasks (Small Task #1 and #2) in the [Large-Scale Multilingual Machine Translation](https://www.statmt.org/wmt21/large-scale-multilingual-translation-task.html) challenge. (See [leaderboard](https://dynabench.org/flores))
* We explored various methods for multilingual machine translation, including sampling methods for back-translation, vocabulary size and the amount of synthetic data.

## What's New
* [01/07/2022] Release reproduction scripts for Small Task #2 (We use almost the same scripts for #1). Note: Currently, we don't have a plan to release the pretrained models.

## Quick Links

  - [Installation](#installation)
  - [Preparation](#preparation)
  - [Train on Parallel Data](#train-on-parallel-data)
  - [Citation](#citation)


## Installation
The installation instruction borrowed from [fairseq](https://github.com/facebookresearch/fairseq). In case of version problem, we offer the fairseq we trained with.
* Clone our repository
  ```bash
  git clone https://github.com/BaohaoLiao/multiback.git
  cd multiback
  ```
* PyTorch >= 1.5.0
* Python >= 3.6
* Install fairseq and develop locally
  ```bash
  git clone https://github.com/pytorch/fairseq
  cd fairseq
  pip install --editable ./
  cd ..
  ```
* For faster training, install NVIDIA's apex:
  ```bash
  git clone https://github.com/NVIDIA/apex
  cd apex
  pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" \
  --global-option="--deprecated_fused_adam" --global-option="--xentropy" \
  --global-option="--fast_multihead_attn" ./
  ```
  
## Preparation
All data and pretrained models are available in the [challenge page](https://www.statmt.org/wmt21/large-scale-multilingual-translation-task.html). 
We mainly show how to process the data of small task #2. For the data of small task #1, just modify the lines with "# TODO" in the scripts for small task #2.
* Download pretrained models.
  ```bash
  mkdir pretrained_models
  cd pretrained_models
  wget https://dl.fbaipublicfiles.com/flores101/pretrained_models/flores101_mm100_615M.tar.gz 
  tar -zxvf flores101_mm100_615M.tar.gz
  rm flores101_mm100_615M.tar.gz
  cd ..
  ```
  |   Pretrained Model (name in our paper) | Original Name | Download |
  |:----------------------:|:--------------:|:---------:|
  | Trans_small |  flores101_mm100_175M | https://dl.fbaipublicfiles.com/flores101/pretrained_models/flores101_mm100_175M.tar.gz |
  | Trans_base | flores101_mm100_615M | https://dl.fbaipublicfiles.com/flores101/pretrained_models/flores101_mm100_615M.tar.gz |
  | Trans_big | m2m_100 |  https://dl.fbaipublicfiles.com/m2m_100/1.2B_last_checkpoint.pt |
* Download datasets.
  ```bash
  mkdir data
  cd data
  # evaluation set
  wget https://dl.fbaipublicfiles.com/flores101/dataset/flores101_dataset.tar.gz
  tar -xzvf flores101_dataset.tar.gz
  rm flores101_dataset.tar.gz
  
  # training set
  wget https://data.statmt.org/wmt21/multilingual-task/small_task2_filt_v2.tar.gz
  tar -xzvf small_task2_filt_v2.tar.gz
  rm small_task2_filt_v2.tar.gz
  cd ..
  ```
* Process parallel data
  ```bash
  cd data_scripts
  # process evaluation set
  bash processEvaluationSetForSmallTask2.sh
  
  # process training set
  python concatenate.py # Concatenate the files with the same translation directions
  bash processTrainSetForSmallTask2.sh
  cd ..
  ```
  Note: For Trans_big, you need to process data like https://github.com/facebookresearch/fairseq/tree/main/examples/m2m_100
  
## Train on Parallel Data
All training scripts are in train_scripts
```bash
cd train_scripts
bash transBaseForSmallTask2ParallelData.sh
```
Here we list the number of GPUs used for each script. If you don't have enough GPUs, just change the flag --update-freq to match our setting.

| Task | Model | Script | #GPU | 
|:-------:|:--------:|:-------:|:--------:|
| Small Task #2 | Trans_small |  transSmallForSmallTask2ParallelData.sh | 32 |
| Small Task #2 | Trans_base  | transBaseForSmallTask2ParallelData.sh |  32 |
| Small Task #2 | Trans_big  | transBigForSmallTask2ParallelData.sh | 128 |

## Back-translation
* You can download the monolingual data [here](https://data.statmt.org/wmt21/multilingual-task/). We don't recommend to use all monolingual data (See Figure 1 in the paper).
* Process monolingual data and generate synthetic data by following https://github.com/facebookresearch/fairseq/tree/main/examples/backtranslation. All three sampling methods are also shown there.
* Combine the parallel data and synthesized data together and retrain the model with above scripts for extra one epoch.

## Citation
Please cite as:
```bibtex
@inproceedings{liao-etal-2021-back,
    title = "Back-translation for Large-Scale Multilingual Machine Translation",
    author = "Liao, Baohao  and
      Khadivi, Shahram  and
      Hewavitharana, Sanjika",
    booktitle = "Proceedings of the Sixth Conference on Machine Translation",
    month = nov,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.wmt-1.50",
    pages = "418--424",
}
```
