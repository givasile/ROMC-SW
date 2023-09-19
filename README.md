# An Extendable Python Implementation of Robust Optimisation Monte Carlo

Paper accepted at the [Journal of Statistical Software (JSS)](https://www.jstatsoft.org/index)

The implemented method can be used through the [ELFI](https://elfi.readthedocs.io/en/latest/) package.


## Authors

* [Vasilis Gkolemis](https://givasile.github.io/)
* [Michael Gutmann](https://michaelgutmann.github.io/)
* [Henri Pesonen](https://scholar.google.fi/citations?user=QS3yn7gAAAAJ&hl=en)

## Abstract

Performing inference in statistical models with an intractable likelihood is challenging, therefore, most likelihood-free inference (LFI) methods encounter accuracy and efficiency limitations. In this paper, we present the implementation of the LFI method Robust Optimisation Monte Carlo (ROMC) in the Python package ELFI. ROMC is a novel and efficient (highly-parallelizable) LFI framework that provides accurate weighted samples from the posterior. Our implementation can be used in two ways. First, a scientist may use it as an out-of-the-box LFI algorithm; we provide an easy-to-use API harmonized with the principles of ELFI, enabling effortless comparisons with the rest of the methods included in the package. Additionally, we have carefully split ROMC into isolated components for supporting extensibility. A researcher may experiment with novel method(s) for solving part(s) of ROMC without reimplementing everything from scratch. In both scenarios, the ROMC parts can run in a fully-parallelized manner, exploiting all CPU cores. We also provide helpful functionalities for (i) inspecting the inference process and (ii) evaluating the obtained samples. Finally, we test the robustness of our implementation on some typical LFI examples.





