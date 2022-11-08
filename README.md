# inverse_design

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

Open ```demo_design_optimization.ipynb``` and run all cells.

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
