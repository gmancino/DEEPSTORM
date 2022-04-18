# Detailed list of instructions to install requirements for experiments
---

## 1. Install Conda

Follow the instructions on the [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) webpage for installation. This project uses Miniconda3 for Linux, e.g.:

```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-ppc64le.sh
chmod +x Miniconda3-latest-Linux-ppc64le.sh
bash Miniconda3-latest-Linux-ppc64le.sh
```

Create a new environment with Python 3.7.9:

```
conda create -n <environment_name> python=3.7.9
conda activate <environment_name>
```

Install `pip` for subsequent steps:

```
conda install -c anaconda pip
```

## 2. Install PyTorch

Follow the instructions on the [PyTorch](https://pytorch.org/get-started/previous-versions/) webpage for installation. Wince we use version 1.6.0 with CUDA version 10.2, use:

```
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2 -c pytorch
```

## 3. Install mpi4py

To perform neighbor communication, install `gcc` and [mpi4py](https://mpi4py.readthedocs.io/en/stable/install.html):

```
conda install -c anaconda gcc_linux-64
conda install -c conda-forge mpi4py openmpi
```

## 4. Install other dependencies

Use `pip install requirements.txt` with the `requirements.txt` file found in this repository.
