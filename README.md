# Public release of the NeuralIL software used in "Neural-network-enabled molecular dynamics study of HfO<sub>2</sub> phase transitions."

## General description
In terms of features, this release is documented in reference [[1]](#1) (see the `uncertainties_with_neuralil` branch of this repository for comparison). However, some classes have a different name and there are minor implementation differences. This specific version was created to ensure total reproducibility presented in "Neural-network-enabled molecular dynamics study of HfO<sub>2</sub> phase transitions."

## Installation

To avoid dependency problems, we suggest the following installation procedure:

1. Create a new virtual environment using, for instance, `conda` or `venv`

2. Install `jax` and `jaxlib`. This can be done, e.g., through the command

```shell
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
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
