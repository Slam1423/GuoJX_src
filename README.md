# GuoJX_src

This repository stores all the needed files(from datasets to source codes) to train the model. 

## Training step

First, change to this directory. Then there are 2 steps to start training after you download this repository as follows.

step1:

```bash
unzip data_cache.zip
```

step2:

```bash
python train.py
```

There's 30 training epoches and the result will be obtained in 0.5-1h with GPU(1080Ti).

## Results

The SOTA's result is [ACC:0.657, Recall:0.611, Precision:0.311, F:0.414].

We get obvious improvement with [ACC:0.751, Recall:0.644, Precision:0.327, F:0.434] as the training result shows.
