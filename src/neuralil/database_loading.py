# Copyright 2019-2021 The NeuralIL contributors
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

import json
import lzma
import random

import numpy as np

from .bessel_descriptors import PowerSpectrumGenerator


class HafniaDatabase:
    def __init__(self, database_location, phase):
        contents = json.load(lzma.open(database_location, mode="rt"))
        indices = [i for i, p in enumerate(contents["phase"]) if p == phase]
        self.chemical_symbols = []
        self.cell = []
        self.positions = []
        self.element_ids = []
        self.energy = []
        self.forces = []
        for i in indices:
            self.cell.append(np.array(contents["cell"][i]))
            self.positions.append(np.array(contents["positions"][i]))
            self.energy.append(contents["energy"][i])
            self.forces.append(np.array(contents["forces"][i]))
            element_ids = []
            for e in contents["elements"][i]:
                if e not in self.chemical_symbols:
                    self.chemical_symbols.append(e)
                element_ids.append(self.chemical_symbols.index(e))
            self.element_ids.append(np.array(element_ids))

    def __len__(self):
        return len(self.element_ids)

    def shuffle(self):
        order = list(range(len(self)))
        random.shuffle(order)
        self.cell = [self.cell[i] for i in order]
        self.positions = [self.positions[i] for i in order]
        self.element_ids = [self.element_ids[i] for i in order]
        self.energy = [self.energy[i] for i in order]
        self.forces = [self.forces[i] for i in order]
        self.element_ids = [self.element_ids[i] for i in order]
