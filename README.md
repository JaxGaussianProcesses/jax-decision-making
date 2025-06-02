# jax-decision-making

[![PyPI - Version](https://img.shields.io/pypi/v/jax-decision-making.svg)](https://pypi.org/project/jax-decision-making)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/jax-decision-making.svg)](https://pypi.org/project/jax-decision-making)

`jax-decision-making` is an early-stage library which aims to provide algorithms for a 
variety of sequential decision-making problems. It currently provides implementations 
of several acquisition/utility functions for Bayesian optimisation, including 
probability of improvement, expected improvement and Thompson sampling. The 
implementations are built upon the JAX library, enabling automatic differentiation, 
vectorisation, and just-in-time (JIT) compilation for high performance on CPUs, GPUs, and TPUs. This allows for efficient research, development, and deployment of decision-making agents.

Initially, `jax-decision-making` was created as a sub-package within the [GPJax](https://github.com/JaxGaussianProcesses/GPJax) library, but it has now been separated 
into its own package. Currently, Gaussian processes (implemented in GPJax) are the 
primary surrogate model around which the library has been developed. Nonetheless, now 
that the packages have been decoupled, we are happy to increase support for alternative
surrogate models, such as Bayesian neural networks etc. Please feel free to open issues 
for features you would like to see implemented. Thise might include:
- Support for alternative sequential decision-making paradigms (e.g. reinforcement 
  learning).
- Support for additional acquisition functions/tricks for Bayesian optimisation and 
  experimental design (e.g. trust-regions for high-dimensional problems).
- Support for alternative surrogate models beyond Gaussian processes (e.g. Bayesian 
  neural networks).

-----

## Table of Contents

- [jax-decision-making](#jax-decision-making)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [License](#license)

## Installation

```console
pip install jax-decision-making
```

## License

`jax-decision-making` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
