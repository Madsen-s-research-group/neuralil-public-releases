# Uncertainty estimation based on NeuralIL

# General description
This is the code associated to the manuscript 

*Deep Ensembles vs. Committees for Uncertainty Estimation in Neural-Network Force Fields: Comparison and Application to Active Learning*

along with the necessary auxiliary files. The core of the code is an updated version of NeuralIL, an automatically differentiable neural-network-based force field presented in Ref.[[1]](#1). Among many other improvements in terms of features and implementation, this version includes a ResNet-inspired core and uses the [VeLO](https://arxiv.org/abs/2211.09760) nonlinear learned optimizer to speed up training by orders of magnitude.

The uncertainty estimators implemented around this core are committees of NeuralIL models and deep ensembles of multi-head neural networks trained using a heteroscedastic loss.

We include data for the ionic liquid ethylammonium nitrate (EAN) and for the perovskite SrTiO3. The former is a superset of the data used in the original NeuralIL article [[1]](#1). The latter combines configurations from Ref.[[2]](#2) with others generated explicitly for this example.

## Installation

At the time of writing (2023-02-09) installing VeLO requires quite specific versions of several libraries. To avoid dependency problems, we suggest the following installation procedure:

1. Create a new virtual environment using, for instance, `conda` or `venv`

2. Install the required versions of `jax` and `jaxlib`. This can be done through the command

```
pip install --upgrade "jax[cuda]==0.3.21" "jaxlib==0.3.20" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

to get the cuda-enabled versions. Remove `[cuda]` to get a CPU-only version.

3. Clone the VeLO repository and install the module

```
git clone git@github.com:google/learned_optimization.git
cd learned_optimization
pip install -e .
```

4. Install the version of NeuralIL distributed with this code by running `pip install -e .`

## Example scripts included

We include a small selection of scripts illustrating training and inference for different types of models:

- `examples/train_an_nn.py`: Train a single NeuralIL model with a ResNet core on data for EAN.
- `examples/train_a_committee.py`: Train a committee of NeuralIL models on data for SrTiO3.
- `examples/train_a_deep_ensemble.py`: Train a deep ensemble of multi-head NeuralIL models on data for EAN.
- `examples/validate_deep_ensemble.py`: Extract statistics from a deep ensemble trained on data for EAN.
- `examples/md_with_committee.py`: Run a short molecular-dynamics (MD) trajectory using a committee of neural networks trained on EAN data.
- `examples/md_with_deep_ensemble.py`: Run a short MD trajectory using a deep ensemble of models trained on EAN data.
- `examples/adversarial_powell.py`: Use adversarial active learning to generate a set of high-uncertainty but plausible surface configurations based on the bulk-only STO potential.

## Data included


- `auxiliary_files/`:
    - `EAN_md_initial.gro`: Initial conditions for the example EAN MD runs.
    - `EAN_params_committee_1FEB3E08.pkl`: Flax parameters for a committee of neural networks trained on EAN data, used for MD.
    - `EAN_params_deep_ensemble_8ABE86C9.pkl`: Flax parameters for a deep ensemble trained on EAN data, used for MD.
    - `STO_params_committee_E5618B21.pkl`: Flax parameters for a committee of neural networks trained on SrTiO3 data, used for active learning.
- `data_sets/`:
    - `EAN/`:
        - `training_original.json`: Training data for EAN, from the original NeuralIL paper [[1]](#1).
        - `validation_original.json`: Validation data for EAN, from the original NeuralIL paper  [[1]](#1).
    - `STO/`:
        - `sto_bulk.json`: Training and validation data for bulk SrTiO3, used in the example training script. Data generated with T=500 K and with T=1000 K are contained in the same file, but tagged with a field "T".
        - `sto_bulk_test.json`: Test data for bulk SrTiO3. Data generated with T=500 K and with T=1000 K are contained in the same file, but tagged with a field "T". 
        - `test_4x1_cells.npy`: Simulation cell sizes for the configurations of the reconstructed 4x1 SrTiO3 surface. This data corresponds to the original SrTiO3 manuscript [[2]](#2).
        - `test_4x1_energies.npy`: Potential energies for the configurations of the reconstructed 4x1 SrTiO3 surface. This data corresponds to the original SrTiO3 manuscript [[2]](#2).
        - `test_4x1_forces.npy`: Forces on atoms for the configurations of the reconstructed 4x1 SrTiO3 surface. This data corresponds to the original SrTiO3 manuscript [[2]](#2).
        - `test_4x1_positions.npy`: Atomic positions for the configurations of the reconstructed 4x1 SrTiO3 surface. This data corresponds to the original SrTiO3 manuscript [[2]](#2).
        - `test_4x1_types.npy`: Atom types for the configurations of the reconstructed 4x1 SrTiO3 surface. This data corresponds to the original SrTiO3 manuscript [[2]](#2).

## References
- <a id="1">[1]</a>
Hadrián Montes-Campos, Jesús Carrete, Sebastian Bichelmaier, Luis M. Varela & Georg K. H. Madsen. *A Differentiable Neural-Network Force Field for Ionic Liquids*. Journal of Chemical Information and Modeling 62 (2022) 88–101. DOI: [10.1021/acs.jcim.1c01380](https://doi.org/10.1021/acs.jcim.1c01380)
- <a id="2">[2]</a>
Ralf Wanzenböck, Marco Arrigoni, Sebastian Bichelmaier, Florian Buchner, Jesús Carrete & Georg K. H. Madsen. *Neural-Network-Backed Evolutionary Search for SrTiO3(110) Surface Reconstructions*. Digital Discovery 5 (2022) 703-710. DOI: [10.1039/D2DD00072E](https://doi.org/10.1039/D2DD00072E)
