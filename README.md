# GEqTrain
GEqTrain if a framework to dynamically build, train and deploy Equivariant Graph based models using pytorch and e3nn

Acknowledgments
---------------

GEqTrain is based on the work of [`nequip`](https://github.com/mir-group/nequip) [1] by mir-group and takes advantage
of the excellent [`e3nn`](https://github.com/e3nn/e3nn) general framework for building E(3)-equivariant neural networks [2,3].

We have adapted the code to work on generic Equivariant Graphs, detaching from NequIP original purpose of learning interatomic potentials.
To improve generalizability and to simplify installation, we include and modify here a subset of `nequip` that is neccessary for our code.

We are grateful for their contribution to the open-source community.

  [1] Batzner, S., Musaelian, A., Sun, L. et al. E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials. Nat Commun 13, 2453 (2022). https://doi.org/10.1038/s41467-022-29939-5
  [2] https://e3nn.org/
  [3] https://zenodo.org/records/7430260