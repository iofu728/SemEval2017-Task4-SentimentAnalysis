# SemEval2017-task4 _Sentiment Analysis in Twitter_

A PKU course project based on the "SemEval-2017 task 4 Sentiment Analysis in Twitter SubTask A" competition.

Classify Problem:

- positive
- negative
- neutral sentiment

## Data info

have emoji(maybe very important)

| -     | pos.         | neu.         | neg.        |
| ----- | ------------ | ------------ | ----------- |
| Train | 19658/39.65% | 22190/44.76% | 7722/15.58% |
| Test  | 5937/48.33%  | 3972/32.33%  | 2375/19.33% |

## Paper reading

## Naive Idea

1. Train textCNN using external dataSet -> embedding, only using A Data, no person info
2. Using bert to do classify

## Baseline

### pre-training embedding

| model | embedding | text_process | r     | p     | f1    | acc   | pos.  | neu.  | neg.  |
| ----- | --------- | ------------ | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| LR    | no        | no           | 45.70 | 43.12 | 44.38 | 48.18 | 43.03 | 74.68 | 11.66 |
| LR    | word2Vec  | no           | 44.87 | 42.48 | 43.65 | 51.37 | 60.81 | 57.27 | 9.37  |
| LR    | fastText  | no           | 43.87 | 42.04 | 42.93 | 46.60 | 44.97 | 72.93 | 8.21  |
| LR    | no        | ekphrasis    | 61.07 | 62.15 | 61.61 | 62.34 | 64.00 | 64.83 | 57.63 |
| LR    | word2vec  | ekphrasis    | 61.49 | 62.35 | 61.92 | 64.37 | 62.52 | 68.72 | 55.80 |
| LR    | fastText  | ekphrasis    | 62.67 | 64.07 | 63.36 | 63.81 | 63.03 | 61.58 | 67.60 |
| Bert  | no        | no           | 20.55 | 33.05 | 25.34 | 22.12 | 88.97 | 10.17 | 0.00  |
| Bert  | no        | ekphrasis    | 38.25 | 33.75 | 35.86 | 48.12 | 1.26  | 97.17 | 2.82  |

## Evaluation

### text_processor = 'no'

| model   | embedding | pad       | r     | p     | f1    | acc   | pos.  | neu.  | neg.  |
| ------- | --------- | --------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| textCNN | fatText   | before 0  | 61.39 | 49.50 | 54.81 | 57.57 | 41.76 | 85.10 | 21.64 |
| textCNN | fastText  | after 1   | 64.77 | 51.93 | 57.64 | 61.10 | 53.97 | 82.90 | 18.91 |
| textCNN | fastText  | before -1 | 63.08 | 52.18 | 57.11 | 59.20 | 45.72 | 83.17 | 27.64 |
| textCNN | fastText  | after end | 62.05 | 57.54 | 59.71 | 63.00 | 57.58 | 76.85 | 38.19 |

### text_processor = 'ekphrasis'

| model   | embedding | pad       | r     | p     | f1    | acc   | pos.  | neu.  | neg.  |
| ------- | --------- | --------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| textCNN | fatText   | before 0  | 61.61 | 65.62 | 63.55 | 63.70 | 69.12 | 56.54 | 71.21 |
| textCNN | fastText  | after 1   | 66.70 | 59.73 | 63.02 | 66.27 | 60.88 | 80.92 | 37.37 |
| textCNN | fastText  | before -1 | 64.71 | 61.61 | 63.12 | 65.23 | 77.62 | 61.70 | 45.51 |
| textCNN | fastText  | after end | 65.36 | 62.37 | 63.83 | 66.93 | 70.02 | 71.12 | 45.98 |

### bert

| model | embedding | text_process | r     | p     | f1    | acc   | pos.  | neu.  | neg.  |
| ----- | --------- | ------------ | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| Bert  | no        | no           | 68.52 | 70.54 | 69.52 | 69.48 | 73.39 | 66.21 | 72.03 |
| Bert  | no        | ekphrasis    | 69.32 | 70.51 | 69.91 | 70.03 | 71.75 | 68.52 | 71.27 |

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

### [OOM] ResourceExhaustedError (see above for traceback): file_name; Disk quota exceeded

Your Disk is full of data.

### INFO:tensorflow:Error recorded from training_loop: corrupted record at 16777214

tf record memory overflow. This situation happen on train time.

[DataLossError (see above for traceback): corrupted record at 12](https://github.com/tensorflow/tensorflow/issues/13463)

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

