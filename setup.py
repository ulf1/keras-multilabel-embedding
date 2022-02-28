import setuptools
import os


def read(fname):
    with open(os.path.join(os.path.dirname(__file__), fname)) as fp:
        s = fp.read()
    return s


def get_version(path):
    with open(path, "r") as fp:
        lines = fp.read()
    for line in lines.split("\n"):
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")


setuptools.setup(
    name='keras-multilabel-embedding',
    version=get_version("keras_multilabel_embedding/__init__.py"),
    description=(
        "Training of multi-label embeddings for k-shingled input sequences."
        " for Tensorflow2/Keras"),
    long_description=read('README.rst'),
    url='http://github.com/ulf1/keras-multilabel-embedding',
    author='Ulf Hamster',
    author_email='554c46@gmail.com',
    license='Apache License 2.0',
    packages=['keras_multilabel_embedding'],
    install_requires=[
        'tensorflow>=2.6.2,<3'
    ],
    python_requires='>=3.6',
    zip_safe=True
)
