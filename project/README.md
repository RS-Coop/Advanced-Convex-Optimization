# Approximate Hessian Based ARC for Non-Convex Optimization
Course project for APPM 5630 - Advanced Convex Optimization, Spring 2021.

## Abstract
We investigate ARC -- a second-order optimization scheme -- with sub-sampling in the Hessian. We then run a number of experiments on deep learning problems.

## Project Members
- Cooper Simpson
- Jaden Wang

## Repository Structure
- *core*: Contains optimizer implementations, deep learning problem setups, and neural network models.
  - *models*: Models used in the various problems we consider
  - *problems*: Image classification problems using CIFAR10 and MNIST datasets
  - *optimizer*: PyTorch optimizer that implements ARC
- *experiments*: Scripts for running problems, plotting, and in general to reproduce our results.
  - *cifar10*
  - *mnist*

## Requirements
- numpy
- scipy
- pytorch
- torchvision
- matplotlib
- seaborn

# Resources and References
There is a large amount of prior research on this subject that has inspired our work and provided invaluable reference. These sources have been organized below by publication date, authors, and material. Links to papers and code repositories have been provided when possible.

- Introduction of Adaptive Cubic Regularization (ARC) methods. The first two papers consider unconstrained optimization of convex problems, while the third extends the work to include non-convex problems.
  - Adaptive Cubic Regularization Methods for Unconstrained Optimization [Part 1](https://link.springer.com/content/pdf/10.1007/s10107-009-0286-5.pdf), [Part 2](https://link.springer.com/content/pdf/10.1007/s10107-009-0337-y.pdf)
  - [An Adaptive Cubic Regularization Algorithm for Non-Convex Optimization](https://people.maths.ox.ac.uk/cartis/papers/cgt32.pdf)

- As stated by the authors this appears to be the first work with strong results on sub-sampling in ARC for non-convex optimization. They also specifically consider the application to machine learning via finite-sum minimization.
  - [Sub-Sampled Cubic Regularization for Non-Convex Optimization](https://arxiv.org/abs/1705.05933)
    - [GitHub Repository](https://github.com/dalab/subsampled_cubic_regularization)

- The following three papers come from a similar set of authors and were the original inspiration for our work. All three deal with non-convex optimization through Newton-type methods (e.g. ARC) using approximate hessian information via sub-sampling. The first paper establishes the theory for their approach with the second being more of a numerical study. The third paper introduces inexact gradient information in addition to sub-sampled hessian.
  - [Newton-Type Methods for Non-Convex Optimization Under Inexact Hessian Information](https://arxiv.org/abs/1708.07164)
  - [Second-Order Optimization for Non-Convex Machine Learning: An Empirical Study](https://arxiv.org/abs/1708.07827)
    - [GitHub Repository](https://github.com/git-xp/Non-Convex-Newton)
  - [Inexact Non-Convex Newton-Type Methods](https://arxiv.org/abs/1802.06925)
    - [GitHub Repository](https://github.com/yaozhewei/Inexact_Newton_Method)

- In this paper a dynamic version of ARC is introduced where the degree of accuracy in the hessian sub-sampling changes adaptively.
  - [Adaptive Cubic Regularization Methods with Inexact Hessian](https://arxiv.org/abs/1808.06239)

- This work introduces a software tool for extracting hessian information from a neural network.
  - [PyHessian: Neural Networks Through the Lens of the Hessian](https://arxiv.org/abs/1912.07145)


## Other References
- [Numerical Optimization by Nocedal and Wright](https://link.springer.com/book/10.1007%2F978-0-387-40065-5)
- [Jadens Original ARC Implementation](https://github.com/tholdem/MatrixMultiplication/blob/master/CubicRegularization/cubicReg.m)
- [Somebodys Git Repo for ARC](https://github.com/cjones6/cubic_reg)
- [PyTorch Optimizer Documentation](https://pytorch.org/docs/stable/optim.html)
