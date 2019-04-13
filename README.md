# SemEval2017-task4 _Sentiment Analysis in Twitter_

A PKU course project based on the "SemEval-2017 task 4 Sentiment Analysis in Twitter SubTask A" competition.

Classify Problem:

- positive
- negative
- neutral sentiment

## Data info

have emoji(maybe very important)

## Paper reading

## Naive Idea

1. Train textCNN using external dataSet -> embedding, only using A Data, no person info
2. Using bert to do classify

## Baseline

| model | embedding | text_process | r     | p     | f1    | acc   | pos.  | neu.  | neg.  |
| ----- | --------- | ------------ | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| LR    | no        | no           | 45.70 | 43.12 | 44.38 | 48.18 | 43.03 | 74.68 | 11.66 |
| LR    | fastText  | no           | 43.87 | 42.04 | 42.93 | 46.60 | 44.97 | 72.93 | 8.21  |
| LR    | no        | ekphrasis    | 61.07 | 62.15 | 61.61 | 62.34 | 64.00 | 64.83 | 57.63 |
| LR    | fastText  | ekphrasis    | 62.67 | 64.07 | 63.36 | 63.81 | 63.03 | 61.58 | 67.60 |

## Trouble Shooting

### ImportError: numpy.core.multiarray failed to import

[numpy version error](https://stackoverflow.com/questions/20518632/importerror-numpy-core-multiarray-failed-to-import)

```sh
$ pip install -U numpy
```

### Can't find module in other folder

colfax import file in sub-folder, so you only can import the file in execute.

### numba.errors.LoweringError: Failed at object (object mode frontend)

numba garbage collect the params which you use once.

so don't use temp_param outer of loop.

```python
@jit
def fastF1(result, predict):
    ''' f1 score '''
    true_total, r_total, p_total, p, r = 0, 0, 0, 0, 0
    for trueValue in range(num_class):
        trueNum, recallNum, precisionNum = 0, 0, 0
    accuracy = trueNum / len(result)
```

is wrong.

### About pickle

before, I dispose pickle big file by byte. But this function can't be use in one time.

It will throw exception like `EOFError: Ran out of input` or `_pickle.UnpicklingError: pickle data was truncated`.

But in fact, It is cause by byte concurrent. It is multiprocessing problem. It's difficult to deal.

```python
def load_bigger(input_file):
    """
    pickle.load big file which size more than 4GB
    """
    max_bytes = 2**31 - 1
    bytes_in = bytearray(0)
    input_size = os.path.getsize(input_file)
    with open(input_file, 'rb') as f_in:
        for _ in range(0, input_size, max_bytes):
            bytes_in += f_in.read(max_bytes)
    return pickle.loads(bytes_in)
```

## Some py skill

### glob -> to match pattern file list

[document](https://docs.python.org/3/library/glob.html)

```python
glob.glob('data/Subtask_A/downloaded/*.tsv')
```

### sklearn pipeline

### print(...., end=' ')

using blank instead of \n in the end

```python
from __future__ import print_function
```
