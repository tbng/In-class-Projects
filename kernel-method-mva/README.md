# Instructions

__Note:__ The Cython code can be viewed easily (similar to Python code) by
openning *.pyx file instead of *.C file.

Step 0. This step might be optional but it is better to run first to install the
requirement libraries:

```python
pip install -r requirements.txt
```

Step 1. First you have to compiled the Cython function (important):

```python
 python3 setup_cython_package.py build_ext --inplace
```
which requires the library `Cython` installed (see step 0).

Step 2a. Then you will be able to run the main file `train.py`

```python
python3 train.py
```

Step 2b. But because this file is training on a whole dataset so it is more reasonable to
run the `example.py` which only run model fitting on a very small proportion of
train and test data set to see if the training process work or not:

```python
python3 example.py
```