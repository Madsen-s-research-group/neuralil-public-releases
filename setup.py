# Copyright 2019-2023 The NeuralIL contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from setuptools import find_packages, setup

setup(
    name="NeuralIL",
    version="0.6.uncertainties",
    description="A Differentiable Neural-Network Force Field",
    author="The NeuralIL contributors",
    author_email="jesus.carrete.montana@tuwien.ac.at",
    package_dir={"": str("src")},
    python_requires=">=3.8",
    packages=find_packages(where="./src"),
    install_requires=[
        "numpy",
        "scipy",
        "tqdm",
        "colorama",
        "matplotlib",
        "ase",
        "jax==0.3.21",
        "jaxlib==0.3.20",
        "flax==0.6.3",
        "optax",
        "learned_optimization",
        "h5py",
        "MDAnalysis",
        "jax_md>=0.2.5",
    ],
)
