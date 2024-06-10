# AICUP Baseline: BoT-SORT ML_10
The code is based on [AICUP_Baseline_BoT-SORT](https://github.com/ricky-696/AICUP_Baseline_BoT-SORT)

## Installation

**The code is tested on Windows 11**

## Setup with Conda
**Step 1.** Create Conda environment and install pytorch
```shell
conda create -n botsort python=3.7
cond activate botsort
```
**Step 2.** Install torch and match torchvision from [pytorch.org](https://pytorch.org/get-started/previous-versions/).<br>
The code is tested using torch 1.13.1+cu117 and torchvision=0.14.1+cu117

**Step 3.** **Install numpy first**
```shell
pip install numpy
```

**Step 4.** Install [pytorch=1.13.1+cu117](https://pytorch.org/get-started/previous-versions/)
```shell
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```

**Step 5.** Install 'requirements.txt'
```shell
cd <AICUP_Baseline_BoT-SORT>

pip install -r 'requirements.txt'
```

**Step 6.** Install cython
```shell
pip install cython
```

**Step 7.** Install [pycocotools](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py)
```shell
pip install pycocotools
```

**Step 8.** Install cython_bbox & faiss-cpu
```shell
# Cython-bbox
pip install cython_bbox

# faiss-cpu
pip install faiss-cpu
```

## Perpare ReID Dataset

For training the ReID , detection patches must be generated as follows:

```shell
# For AICUP
python fast_reid/datasets/generate_AICUP_patches.py --data_path D:\Data\train

# remember link dataset to FastReID
set FASTREID_DATASETS=fast_reid\datasets
```
>[!TIP]
>+ Since we previously changed the directory to `AICUP_Baseline_BoT-SORT`, 
> the `FASTREID_DATASETS=fast_reid\datasets` should be set as a relative path.
>+ In the [AICUP_Baseline_BoT-SORT](https://github.com/ricky-696/AICUP_Baseline_BoT-SORT)
> use ```export FASTREID_DATASETS=fast_reid\datasets```, but Windows don't have ''''rgb(255, 0, 0)'export'''
>command, to translate linux style command scipt to windows/command batch style it would go like this:
>```
>set FASTREID_DATASETS=fast_reid\datasets
>```
