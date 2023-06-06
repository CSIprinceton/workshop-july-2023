# Deep Modeling for Molecular Simulation
Hands-on sessions - Day 2 - July 12, 2023

Train your first first-principles machine-learning force field

## Aims

Using the DFT energies and forces obtained in the previous tutorial, train a model for the potential energy surface using DeePMD-kit.

## Objectives

The objectives of this tutorial session are:
- Train a DeePMD model for the potential energy surface of silicon
- Become familiar with the inputs and outputs of the training process
- Run molecular dynamics simulations driven by the DeePMD model
- Identify strengths and limitations of a rudimentary model
- Use an ensemble of models to estimate the errors in the forces

## Prerequisites

It is assumed that the student has completed all hands-on sessions from [day 1](https://github.com/CSIprinceton/workshop-july-2023/tree/main/hands-on-sessions/day-1) of this workshop.

## Training data

We will use the training data for silicon prepared on [hand-on session 3](https://github.com/CSIprinceton/workshop-july-2023/tree/main/hands-on-sessions/day-1/3-preparing-training-data).
These data consist of:
- Around 400 configurations of silicon in the cubic diamond crystal structure obtained using random displacements from equilibrium atomic positions. 
- Around 300 configurations of liquid silicon at 1700 K obtained in molecular dynamics simulations driven by the [Stillinger-Weber potential](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.31.5262).
Energies and forces for these configurations were obtained using DFT with the PBE functional. 

We shall see below whether a DeePMD model trained on the configurations described above is able to drive the dynamics of this system, while preserving a high-accuracy.
Can you guess if the model will be good for the liquid, the solid, none, or both? Why?
Let's make a poll in the classroom!

## Training input file
