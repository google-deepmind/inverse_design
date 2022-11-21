# Inverse Design for fluid-structure interactions using graph network simulators

Code and parameters to accompany the NeurIPS 2022 paper
**Inverse Design for Fluid-Structure Interactions using Graph Network
Simulators** ([arXiv](https://arxiv.org/abs/2202.00728))<br/>
_Kelsey R. Allen*, Tatiana Lopez-Guevara*, Kimberly Stachenfeld*,
Alvaro Sanchez-Gonzalez, Peter Battaglia, Jessica Hamrick, Tobias Pfaff_

The code here provides an implementation of the Encode-Process-Decode
graph network architecture in jax, model weights for this architecture trained
on the 3D WaterCourse environment, and an example of performing gradient-based
optimization in order to optimize a landscape to reroute water.

## Usage

### in a google colab
Open the [google colab](https://colab.research.google.com/github/deepmind/inverse_design/blob/master/demo_design_optimization.ipynb) and run all cells.

### with jupyter notebook / locally
To install the necessary requirements (run these commands from the directory
that you wish to clone `inverse_design` into):

```shell
git clone https://github.com/deepmind/inverse_design.git
python3 -m venv id_venv
source id_venv/bin/activate
pip install --upgrade pip
pip install -r ./inverse_design/requirements.txt
```

Additionally install jupyter notebook if not already installed with
`pip install notebook`

Finally, make a new directory within the `inverse_design` repository and move
files there:
```shell
cd inverse_design
mkdir inverse_design
mv src/ inverse_design/
```

Download the dataset and model weights from google cloud:
```shell
wget -O ./gns_params.pickle https://storage.googleapis.com/dm_inverse_design_watercourse/gns_params.pickle
wget -O ./init_sequence.pickle https://storage.googleapis.com/dm_inverse_design_watercourse/init_sequence.pickle
```

Now you should be ready to go! Open `demo_design_optimization.ipynb` inside
a jupyter notebook and run *from third cell* onwards.

## Citing this work

If you use this work, please cite the following paper
```
@misc{inversedesign_2022,
  title = {Inverse Design for Fluid-Structure Interactions using Graph Network Simulators},
  author = {Kelsey R. Allen and
               Tatiana Lopez{-}Guevara and
               Kimberly L. Stachenfeld and
               Alvaro Sanchez{-}Gonzalez and
               Peter W. Battaglia and
               Jessica B. Hamrick and
               Tobias Pfaff},
  journal = {Neural Information Processing Systems},
  year = {2022},
}
```
## License and disclaimer

Copyright 2022 DeepMind Technologies Limited

All software is licensed under the Apache License, Version 2.0 (Apache 2.0);
you may not use this file except in compliance with the Apache 2.0 license.
You may obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.
