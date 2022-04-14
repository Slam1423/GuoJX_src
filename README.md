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

There's 30 training epoches and the training will cost 0.5-1 hour with GPU(1080Ti).

### Possible error:

```bash
  File "train.py", line 68
    seq_var = torch.autograd.Variable(seq_data.cuda(async=True).float())
                                                        ^
SyntaxError: invalid syntax
```

This error can be easily addressed by deleting all "async=True" in train.py. Then the training will be OK.

## Results

The SOTA's result is [ACC:0.657, Recall:0.611, Precision:0.311, F:0.414].

We get obvious improvement with [ACC:0.742, Recall:0.639, Precision:0.329, F:0.434] as the training result shows.
