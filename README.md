hyperbolicMDS
&middot;
[![Latest Github release](https://img.shields.io/github/release/enlberman/hyperbolicMDS.svg)](https://github.com/enlberman/hyperbolicMDS/releases/latest)
[![PyPI version](https://img.shields.io/pypi/v/hyperbolicMDS.svg)](https://pypi.python.org/pypi/hyperbolicMDS)
=====

An implementation of multi dimensional scaling in the poincare ball. Adapted from [hyperbolic-learning](https://github.com/drewwilimitis/hyperbolic-learning) code for mds on the unit poincare disk.


Running MDS on an input symmetric and positive distance matrix is simple:
```c
from hyperbolicMDS.mds import HyperMDS

h_mds = HyperMDS(dissimilarity='precomputed')
embedding = h_mds.fit_transform(input_distance_matrix, max_epochs=100000, rmax=rmax, rmin=rmin)
```
Since space expands on the poincare ball as you get farther from the origin, distances are not scale-invariant. As a result, it is critical to set rmax and rmin to values that are appropriate for your data. One way to figure out what these values should be is to use [ALBATROSS](https://github.com/enlberman/albatross).

Once you have an embedding you can compare the distances in hyperbolic space to the input distance matrix:
```c
from hyperbolicMDS.mds import poincare_dist_vec
from scipy.stats import spearmanr

embedding_dists = poincare_dist_vec(embedding)

spearmanr(embedding_dists.flatten(),input_distance_matrix.flatten())
```

This implementation uses a modified version of [ADAM](https://arxiv.org/abs/1412.6980) for gradient descent. If your MDS is not converging quickly over the first epochs you may need to adjust the ADAM parameters. Sometimes, the best results can be achieved by modifying the beta10 and beta20 parameters:
```c
embedding = h_mds.fit_transform(input_distance_matrix, max_epochs=100000, rmax=rmax, rmin=rmin, beta10=.5, beta20=.8)
```

The betas decay with the square root of epochs so smaller parameters will decay faster leading to smaller step sizes faster and larger values will decay more slowly keeping step sizes large initially. 
