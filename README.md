# QMC_RandomSpinChains

This repository contains code to simulate a spin-1/2 random antiferromagnetic Heisenberg chains using the Stochastic Series Expansion algorithm.

## Running the simulation

To run the main part of the code:

```bash
g++ -fopenmp -O3 code/main.cpp -o main && ./main
```
This will produce .txt files with the disorder-averaged zz correlation function.

## Plotting

To plot the results, first install matplotlib and numpy with:

```bash
pip install -r code/requirements.txt
```
And plot using

```bash
python code/plot_results.py
```
## About the algorithm

The Stochastic Series Expansion algorithm is based on a Taylor expansion of the parition function. It is highly efficient and versatile, with typical runtimes scaling linearly with the system size and inverse temperature $O(N\beta)$. The implementation presented here is based on the pseudocode detailed in the lecture notes *Computational Studies of Quantum Spin Systems* by Anders Sandvik (https://doi.org/10.1063/1.3518900). Available at: https://arxiv.org/abs/1101.3281
