# Deep Modeling for Molecular Simulation
Hands-on sessions - Day 2 - July 12, 2023

## Aims

Learning to analyze and understand the prediction errors of the trained model.

## Objectives

This is a short session where we will
- Learn to use Python APIs for model inference
- Compute and analyze the root mean square error
- Make parity plots and error distribution plots

## Prerequisites

It is assumed that the student has finish the active learning [day 1](https://github.com/CSIprinceton/workshop-july-2023/tree/main/hands-on-sessions/day-2/5-active-learning) session and obtained the datasets and trained models.

## Running Jupyter Notebook

The example code is given in the jupyter notebooks ```Analysis.ipynb```, and you can run it and tweek it for further analysis. Jupyter notebooks can be run the virtual machine and open it in your local browser.
In order to do this, first execute on the remote machine:
```
nohup jupyter notebook --port=2333 &
```
and then run in your local machine:
```
ssh -N -f -L localhost:2333:localhost:2333 -p <port> <username>@<remote-machine-address>
```
At the end of the ```nohup.out``` file you will find a link that you can copy and then paste into your browser.

## Computing Mean Square Error

The most imporatant thing to see is the root mean square error. 
The root mean square error for a quantity $X$ with predictions $X_{\pred}$ and ground truth $X_{\true}$ is given by 
$$ RMSE = \langle (X_{\pred} - X_{\true})^2 \rangle^{\frac{1}{2}}. $$
In DP, the training loss is a combination of the square error of energy, force, and sometime virial.
The error for energy and force is usually no more than $1$ meV/atom and $10^{-1} $eV/Å, 
with the specifics depending on the system.  
The model and training hyperparameters can also have an influence, 
see for example [model size](https://github.com/deepmodeling/deepmd-kit/blob/master/doc/troubleshooting/howtoset_netsize.md). 
For typical machine learning tasks where data are drawn from a single distribution, 
you should test the generalization error on a held-out dataset, 
which should not be too large compared to the training error.
However, here we use DP-GEN which gives a series of different distributions, 
so you may check the error on the constituent data systems in the jupyter notebook, 
which is of more interest.

## Parity plots

Also a common type of anaysis is parity plots, 
which is a scatterplot that compares the predicted value against the true value. 
A well-trained model's predictions should not display significant outliers. 
We've plotted the one for the force, and you can try doing the same for the energy.

# Error Distribution

We can also plot the error distribution. Usually they should give an Gaussian-like distribution, but our dataset has several different constituents, which may complicate things. Again, we plot the forces, and you can do the same for the energy, and you can also try it on the different subsets of data.
