***

The directory contains:

- `replication_script.py`: Standalone script that replicates all the examples presented in the article

- `requirements.txt`: Pip requirements for running the replication_script.py.
- `dockerfile.txt`: for building the docker image
***

We propose two approaches for running the `replication_script.py`. During testing we noticed that there is a reproducibility issue related to the Bayesian Optimization execution. Some parts of the underlying GPy library relate to Cython executables which makes the execution OS-sensitive. This means that for the Anaconda-based environment, the Bayesian Optimization part (and only thay) gives slighlty different numbers, if running the script in different PCs . Please, notice, that results are consistent between different runs on the same PC; running `replication_script.py` multiple times will output the exact same numbers.

For this reason, we propose a second approach, creating a docker image and running the script through the image which ensures perfect reproducibility. Bellow, we summarize the two approaches:

1. Install an Anaconda environment:

```
conda create -n romc python=3.9
conda activate romc
pip install --upgrade pip
pip install -r requirements.txt
python replication_script.py
```

2. Install docker (version 24.0.6) and build/run a docker image:

``` shell
docker build -t romc_jss .
docker run -it romc_jss
```

***

The replication script is divided in 5 parts;

- Part 1 generates the subfigures of figure 3 (the running example for understanding the implementation) and prints some statistics of the inference

- Part 2 generates some subfigures of figures 4, 5, 6. It performs inference on the example of section 5, using the gradient-based optimiser

- Part 3 generates some subfigures of figures 4, 5, 6. It performs inference on the example of section 5, using the Bayesian optimisation scheme. It also perfoms inference using Rejection sampling for comparing the results.

- Part 4 generates some subfigures of figures 5, 6. It performs inference on the example of section 5, using the Neural Network extension.

- Part 5 generates the subfigures of figure 7, for measuring/comparing the execution times

***

Comments:

- The `replication_script.py` takes about 25 minutes to run.

***
