# README.md

This repository contains the data and PyTorch implementation of the anonymous 
submission _NegatER: Generating Negatives in Commonsense Knowledge Bases by Mining Language Models._

## Quick start

Run the following to set up your virtual environment and install the Python requirements: 
```
python3.7 -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
```
Since NegatER makes use of the FAISS library, you also need to install libomp and libopenblas.
On Ubuntu, run the following:
```
apt install libopenblas-base libomp-dev
```
On OS X, run the following:
```
brew install libomp openblas
```

## Data

The repository includes both datasets used in our experiments: 

- `data/conceptnet/full`: The __full ConceptNet benchmark__ from [Li et al 2016](https://www.aclweb.org/anthology/P16-1137.pdf),
which comprises 100,000/2,400/2,400 train/validation/test triples across 34 relations and 78,334 unique phrases.
- `data/conceptnet/true-neg`: The __filtered ConceptNet dataset__ consisting of six relations that
have true negative counterparts in the original benchmark. It comprises 36,210/3,278/3,278 train/validation/test triples
across six relations and 41,528 unique phrases. 

## Jobs

Each job (fine-tuning and/or negative generation) requires a YAML configuration file. 
The file `config-default.yaml` provides default configuration options across all jobs,
alongside explanations of each configuration key. 
You can overwrite these options by creating your own config file.

### Language model fine-tuning

To fine-tune a language model on a commonsense KB, use the `src/fine_tune.py` script:
```
usage: fine_tune.py [-h] [--action {train,test} [{train,test} ...]]
                    [--config-file CONFIG_FILE]
                    config_dir

positional arguments:
  config_dir            Directory of job config file

optional arguments:
  -h, --help            show this help message and exit
  --action {train,test} [{train,test} ...]
                        Default: ['train', 'test']
  --config-file CONFIG_FILE
                        Configuration filename in the specified config
                        directory. Default: 'config.yaml'
```

Here are some examples of commands you can run:

- To fine-tune and test BERT-Base for on the __full ConceptNet 
  dataset__ using our given (best) config file, run the following:
  ```
  python src/fine_tune.py configs/conceptnet/full/classify/
  ```
- To _test_ your fine-tuned BERT-Base on the __full ConceptNet dataset__, run the following:
  ```
  python src/fine_tune.py configs/conceptnet/full/classify/ --action test
  ```
- To fine-tune RoBERTa-Base instead of BERT on the __full ConceptNet dataset__,
  edit the `configs/conceptnet/full/classify/config.yaml` file as follows:
  ```
  fine_tune:
    model:
      pretrained_name: roberta-base

    eval:
      test_checkpoint: roberta_best
  ```
  then run the following:
  ```
  python src/fine_tune.py configs/conceptnet/full/classify/
  ```
- To fine-tune BERT with negative samples generated by the UNIFORM baseline on the
 __filtered ConceptNet dataset__, run the following: 
  ```
  python src/fine_tune.py configs/conceptnet/true-neg/uniform/
  ```
- To fine-tune BERT using the negative samples generated by NegatER-$\nabla$ on the __filtered
  ConceptNet dataset__, run the following: 
  ```
  python src/fine_tune.py configs/conceptnet/true-neg/negater-gradients/
  ```

### NegatER generation

To generate negatives given a language model fine-tuned on a commonsense KB,
use the `src/negater.py` script:
```
usage: negater.py [-h] [--type {thresholds,full-gradients,proxy-gradients}]
                  [--config-file CONFIG_FILE]
                  config_dir

positional arguments:
  config_dir            Directory of job config file

optional arguments:
  -h, --help            show this help message and exit
  --type {thresholds,full-gradients,proxy-gradients}
                        Type of NegatER job to run: 'thresholds' for
                        NegatER-$\theta_r$, 'full-gradients' for
                        NegatER-$\nabla$ without the proxy, and 'proxy-
                        gradients' for NegatER-$\nabla$ with the proxy.
                        Default: 'thresholds'
  --config-file CONFIG_FILE
                        Configuration filename in the specified config
                        directory.Default: 'config.yaml'
```
Usage examples:

- To generate negatives with __NegatER-$\theta_r$__ using our precomputed _k_-nearest-neighbors
  index for _k_=10, run the following:
  ```
  python src/negater.py configs/conceptnet/full/generate/ --type thresholds
  ```
- To generate negatives with __NegatER-$\nabla$__ + the proxy approach,
  building and saving a new _k_-nearest-neighbors index for _k_=20, 
  first modify the `configs/conceptnet/full/generate/config.yaml` file as follows:
  ```
  negater:
    index:
      build: True
      k: 20
      save: True
  ```
  then run the following:
  ```
  python src/negater.py configs/conceptnet/full/generate/ --type proxy-gradients
  ```