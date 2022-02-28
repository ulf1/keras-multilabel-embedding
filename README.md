[![PyPI version](https://badge.fury.io/py/keras-multilabel-embedding.svg)](https://badge.fury.io/py/keras-multilabel-embedding)
[![Total alerts](https://img.shields.io/lgtm/alerts/g/ulf1/keras-multilabel-embedding.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/ulf1/keras-multilabel-embedding/alerts/)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/ulf1/keras-multilabel-embedding.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/ulf1/keras-multilabel-embedding/context:python)
[![keras-multilabel-embedding](https://snyk.io/advisor/python/keras-multilabel-embedding/badge.svg)](https://snyk.io/advisor/python/keras-multilabel-embedding)

# keras-multilabel-embedding
The package contains a TensorFlow2/Keras class to train an Embedding matrix for multi-label inputs, i.e. instead of 1 ID per token (one hot encoding), N IDs per token can be provided as model input.


An PyTorch implementation can be found here:
https://github.com/ulf1/torch-multilabel-embedding
(`pip install torch-multilabel-embedding`)

## Usage

### Multi-label embeddings with fixed number of labels
```py
import keras_multilabel_embedding as tml
import tensorflow as tf

# a sequence of multi-label data points
x_ids = [[1, 2, 4], [0, 1, 2], [2, 1, 4], [3, 2, 1]]
x_ids = tf.constant(x_ids)

# initialize layer
layer = tml.MultiLabelEmbedding(
    vocab_size=5, embed_size=300, random_state=42)

# predict
y = layer(x_ids)
```

### Multi-label embeddings with variable number of labels

```py
import keras_multilabel_embedding as tml
import tensorflow as tf

# a sequence of multi-label data points
x_ids = [[1, 2, 4], [0, 1, 2], [2, 1], [3]]

# initialize layer
layer = tml.MultiLabelEmbedding(
    vocab_size=5, embed_size=300, random_state=42)

# predict
y = layer(x_ids)
```


## Appendix

### Installation
The `keras-multilabel-embedding` [git repo](http://github.com/ulf1/keras-multilabel-embedding) is available as [PyPi package](https://pypi.org/project/keras-multilabel-embedding)

```sh
pip install keras-multilabel-embedding
pip install git+ssh://git@github.com/ulf1/keras-multilabel-embedding.git
```

### Install a virtual environment

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir
pip install -r requirements-dev.txt --no-cache-dir
pip install -r requirements-demo.txt --no-cache-dir
```

(If your git repo is stored in a folder with whitespaces, then don't use the subfolder `.venv`. Use an absolute path without whitespaces.)

### Python commands

* Jupyter for the examples: `jupyter lab`
* Check syntax: `flake8 --ignore=F401 --exclude=$(grep -v '^#' .gitignore | xargs | sed -e 's/ /,/g')`
* Run Unit Tests: `PYTHONPATH=. pytest`

Publish

```sh
pandoc README.md --from markdown --to rst -s -o README.rst
python setup.py sdist 
twine upload -r pypi dist/*
```

### Clean up 

```sh
find . -type f -name "*.pyc" | xargs rm
find . -type d -name "__pycache__" | xargs rm -r
rm -r .pytest_cache
rm -r .venv
```


### Support
Please [open an issue](https://github.com/ulf1/keras-multilabel-embedding/issues/new) for support.


### Contributing
Please contribute using [Github Flow](https://guides.github.com/introduction/flow/). Create a branch, add commits, and [open a pull request](https://github.com/ulf1/keras-multilabel-embedding/compare/).
