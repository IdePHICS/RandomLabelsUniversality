# Universality with random labels
Repository for the paper [*Gaussian Universality of Linear Classifiers with Random Labels in High-Dimension*](https://arxiv.org/abs/XXXX.XXXX).

<p float="left">
  <img src="https://github.com/gsicuro/GaussMixtureProject/blob/main/plots/animation_logistic.gif" height="270" />
  <img src="https://github.com/gsicuro/GaussMixtureProject/blob/main/plots/GenErr.jpg" height="270">
  <img src="https://github.com/gsicuro/GaussMixtureProject/blob/main/plots/TrainErr.jpg" height="270">
</p>

*Left: logistic classification of three clusters with ridge regularisation for different values of the regularisation's strength λ. Center and right: test error and training error performing a ridge classification of a mixture of K=3 clusters with diagonal covariance in the high dimensional limit, with thoretical predictions compared with the results of numerical simulations.*

## Structure

In this repository we provide the code and some guided example to help the reader to reproduce the figures of the paper [1]. The repository is structured as follows.

| File                          | Description                                                                                                                                                    |
|-------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ```/K2mix``` | Solver for the fixed point equations of the order parameters in the case of classification tasks on K=2 Gaussian clusters with nonhomogeneous diagonal covariances. The notebook [```how_to.ipynb```](https://github.com/gsicuro/GaussMixtureProject/blob/main/K2mix/how_to.ipynb) provides a step-by-step explanation on how to use the package. This implementation has been used to produce the results in Section 3.1 of the paper.           |
| ```/multiK``` | Solver for the fixed point equations of the order parameters in the case of classification tasks on K Gaussian clusters with diagonal covariances. The notebook [```how_to.ipynb```](https://github.com/gsicuro/GaussMixtureProject/blob/main/multiK/how_to.ipynb) provides a step-by-step explanation on how to use the package. This implementation has been used to produce the results in Section 3.2 of the paper.                                     |

The notebooks are self-explanatory.

## Reference

[1] *Gaussian Universality of Linear Classifiers with Random Labels in High-Dimension*,
Federica Gerace, Florent Krzakala, Bruno Loureiro, Ludovic Stephan, Lenka Zdeborová, [arXiv: XXXX.XXXX](https://arxiv.org/abs/XXXX.XXXX)[stat.ML]
