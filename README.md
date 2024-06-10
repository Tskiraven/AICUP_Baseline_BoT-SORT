# AICUP Baseline: BoT-SORT
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

**Step 4.** Install pytorch=1.13.1+cu117
```shell
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```

**Step 5.** Install 'requirements.txt'
```shell
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