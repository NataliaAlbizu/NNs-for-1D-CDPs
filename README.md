# Overview

This repository contains the code developed for my bachelor's thesis, *"Neural Networks for One-Dimensional Convection-Diffusion Problems."*

# Project Structure

The `neural_networks` folder contains the implementations of the neural networks for the model problem (`fem_nn.py` and `pinn.py`) and the code for the introductory chapter on deep learning.

The `fdm_and_fem` folder contains the implementations for the numerical discretization using the Finite Difference Method (FDM) and the Finite Element Method (FEM).

The `analytical` folder contains notebooks related to the analytical behavior of the model problem.

# References

Some implementations and examples in this repository are based on or inspired by the following references:

1. **Novo, J., & Terrés, E. (2025).** Can neural networks learn finite elements? *Journal of Computational and Applied Mathematics, 453*, 116168.  
   [https://doi.org/10.1016/j.cam.2024.116168](https://doi.org/10.1016/j.cam.2024.116168)  
   *Basis for the neural network implementation in `fem_nn.py`.*

2. **Higham, C. F., & Higham, D. J. (2019).** Deep Learning: An Introduction for Applied Mathematicians. *SIAM Review, 61*(4), 860–891.  
   [https://doi.org/10.1137/18M1165748](https://doi.org/10.1137/18M1165748)  
   *Example reproduced in the `intro_ANN` notebook.*

3. **Nguyen, T. K. (n.d.).** PINNs Tutorial. GitHub repository.  
   [https://github.com/nguyenkhoa0209/pinns_tutorial](https://github.com/nguyenkhoa0209/pinns_tutorial)  
   *Reference for the PINN implementation in `pinn.py`.*

# Author

Natalia Albizu