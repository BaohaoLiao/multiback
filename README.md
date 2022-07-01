# Multilingual Backtranslation
This repository provides scripts, including preprocessing and training, for our WMT21's paper, [Back-translation for Large-Scale Multilingual Machine Translation](https://aclanthology.org/2021.wmt-1.50/).

## Key Points of Our Paper
* We achieved the 2nd space for small tasks (Small Task #1 and #2) in the [Large-Scale Multilingual Machine Translation](https://www.statmt.org/wmt21/large-scale-multilingual-translation-task.html) challenge. (See [leaderboard](https://dynabench.org/flores))
* We explored various methods for multilingual machine translation, including sampling methods for back-translation, vocabulary size and the amount of synthetic data.

## What's New
* [01/07/2022] Release reproduction scripts. Note: Currently, we don't have a plan to release the pretrained models.

## Installation
The installation instruction borrowed from [fairseq](https://github.com/facebookresearch/fairseq)
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
  # https://dl.fbaipublicfiles.com/flores101/pretrained_models/flores101_mm100_175M.tar.gz for flores101_mm100_175M	 
  wget https://dl.fbaipublicfiles.com/flores101/pretrained_models/flores101_mm100_615M.tar.gz 
  tar -zxvf flores101_mm100_615M.tar.gz
  rm flores101_mm100_615M.tar.gz
  cd ..
  ```
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
  bash process_evaluationSet_for_smallTask2.sh
  
  # process training set
  python concatenate.py # Concatenate the files with the same translation directions
  bash process_trainSet_for_smallTask2.sh
  cd ..
  ```
  
## Train Models on Parallel Data
All training scripts are in train_scripts
```bash
cd train_scripts
bash train_615m_for_smallTask2_parallelData.sh
```
Here we list the number of GPUs used for each script.

|              Script              | #GPU | 
|:-------------------------------|:--------:|
| train_615m_for_smallTask2_parallelData.sh |  40 |



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
