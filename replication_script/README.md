***

The directory contains:

- `replication_script.py`: Standalone script that replicates all the examples presented in the article

- `requirements.txt`: Pip requirements for running the replication_script.py.

***

For running the script using `requirements-from-repo.txt`, follow the steps:

- Install `Python 3.7.9` in your system
- Initiate a virtual environment

`python -m venv env`

`source env/bin/activate`

* Install all packages described in requirements.txt:

`pip install --upgrade pip`

`pip install -r requirements.txt`

* Run the script

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

- The figures are stored in both `.png` and `.tex` version. For the latter to work, apart from the python package tikzplotlib, the latest `Tex` environment must be installed. If this is not the case, you may set `savefig=False` at line 20 of the `replication_script.py`.

- The `replication_script.py` takes about 30 minutes to run.

***
