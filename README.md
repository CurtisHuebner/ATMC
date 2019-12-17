# ATMC
This repo implements a slight variation of the ATMC algorithm described in algorithm 1 of "[Bayesian Inference for Large Scale Image Classification](https://arxiv.org/abs/1908.03491)" by Heek et al. 

The implementation subclasses the standard pytorch optimiser class and so can be used as a drop in replacement for your usual optimisation algorithm such as SGD.

To test the algorithm simply run:
```
>>> python3 atmc.py
```
The output should resemble Samples.png, which represents samples taken from a Gaussian distribution and samples taken from the sampler with the Gaussian energy function. This indicates that the sampler is able to match the distribution implied by the energy function.

As a further demonstration, no_randomfit.py shows that the ATMC algorithm with an over parameterized neural network cannot fit random data, while other algorithms will.
```
>>> python3 no_randomfit.py
```

The implementation deviates from what is described in Algorithm 1 of the paper in that the partial sde is not integrated. A linear approximation is used instead. This was done because using the exact integral provided in the paper led to incorrect sampling.
