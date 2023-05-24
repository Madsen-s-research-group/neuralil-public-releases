# Public release of the NeuralIL software for use with Clinamen2

## General description
In terms of features, this release is documented in reference [[1]](#1) (see the `uncertainties_with_neuralil` branch of this repository for comparison). The main differences are:

- Integration with [ASE](https://wiki.fysik.dtu.dk/ase/)
- Some minor additions to the functionality of committees, related to the aforementioned integration
- Compatibility with newer versions of `jax` and `jaxlib`.
- Only a single example dataset (Training data for EAN, from the original NeuralIL paper [[2]](#2)) is included.

## Installation

To avoid dependency problems, we suggest the following installation procedure:

1. Create a new virtual environment using, for instance, `conda` or `venv`

2. Install `jax` and `jaxlib`. This can be done, e.g., through the command

```shell
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

to get a CUDA12-enabled version. See the [JAX installation guide](https://github.com/google/jax#installation) for other options, including CPU-backed installations.

3. Clone the [VeLO](https://arxiv.org/abs/2211.09760) repository and install the module

```shell
git clone git@github.com:google/learned_optimization.git
cd learned_optimization
pip install -e .
```

4. Install this version of NeuralIL by running `pip install -e .`

## References
- <a id="1">[1]</a>
Jesús Carrete, Hadrián Montes-Campos, Ralf Wanzenböck, Esther Heid, Georg K. H. Madsen. *Deep Ensembles vs Committees for Uncertainty Estimation in Neural-Network Force Fields: Comparison and Application to Active Learning*. The Journal of Chemical Physics 158 (2023) 204801. DOI: [10.1063/5.0146905](https://doi.org/10.1063/5.0146905)
- <a id="2">[2]</a>
Hadrián Montes-Campos, Jesús Carrete, Sebastian Bichelmaier, Luis M. Varela & Georg K. H. Madsen. *A Differentiable Neural-Network Force Field for Ionic Liquids*. Journal of Chemical Information and Modeling 62 (2022) 88–101. DOI: [10.1021/acs.jcim.1c01380](https://doi.org/10.1021/acs.jcim.1c01380)
