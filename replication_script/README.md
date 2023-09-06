***

The directory contains:

- `replication_script.py`: Standalone script that replicates all the examples presented in the article

- `requirements.txt`: Pip requirements for running the replication_script.py.

***

For running the script `replication_script.py`, follow the steps:

1. Install `Python 3.9` at your system and initialize a virtual environment

You can use any approach for Step 1. Two easy solutions are (i) using Anaconda 

```
conda create -n romc python=3.9
conda activate romc
pip install --upgrade pip
pip install -r requirements.txt
```

or (ii) download and install `python=3.9` executable and then create a virtual environment using:

```
python -m venv env
source env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

2. Run the script

`python replication_script.py`

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
