# Deep Modeling for Molecular Simulation
Hands-on sessions - Day 4 - July 14, 2023

Run metadynamics simulations to study the liquid-solid transition

## Aims

Use the DeePMD model for silicon trained on [day 2](https://github.com/CSIprinceton/workshop-july-2023/tree/main/hands-on-sessions/day-2) to run metadynamics of the liquid-solid transition with the [PLUMED](https://www.plumed.org/) enhanced sampling plugin.

## Objectives

The objectives of this tutorial session are:
- Setup and run a metadynamics simulation
- Compute free energy profiles along a collective variable
- Understand the notion of unbiased and biased distributions of the collective variable
- Use reweighting to compute unbiased distributions
- Determine thermodynamic conditions for equilibrium between phases
- Compute chemical potential differences

## Prerequisites

It is assumed that the student has completed all hands-on sessions from [day 1](https://github.com/CSIprinceton/workshop-july-2023/tree/main/hands-on-sessions/day-1) and [day 2](https://github.com/CSIprinceton/workshop-july-2023/tree/main/hands-on-sessions/day-2) of this workshop.


## Introduction

During crystallization the disordered atoms of a liquid spontaneously organized into periodic patterns with long range order.
The time and lengthscales involved are often too short to be studied with experiments.
In this tutorial we will see how we can study this fascinating process using enhanced sampling molecular dynamics simulations.
We take as example the case of silicon that crystallizes in the cubic diamond crystal structure.

Below we provide a (very short) summary of the methods that will be employed.

### Well-tempered metadynamics

In well-tempered metadynamics a bias potential $V(s)$ is constructed as a function of some collective variable *s*.
The bias potential is constructed as a sum of repulsive Gaussians that discourage frequently visited configurations.
In this way, the simulation explores different regions of the free energy surface $F(s)$ of the system.
In the long time limit the bias potential converges to,

$V(s)= - \left ( 1- \frac{1}{\gamma} \right ) F(s)$

where $\gamma$ is the bias factor.

The effective free energy $\tilde F(s)$ when the bias is converged will be:

$\tilde F(s) = F(s) / \gamma$

$F(s)$ and $\tilde F(s)$ are connected to the unbiased and biased probability distributions of the CV, and their definitions are:

$F(s) = -k_B T \ln \left( P(s) \right)$

and,

$\tilde F(s) = -k_B T \ln \left( P_B(s) \right)$

where $P(s)$ is the unbiased distribution and $P_B(s)$ is the biased distribution of the CV.


### Collective variable

The collective variable (CV) that we will use is based on comparing the atomic environments in the simulation with those of a reference crystal structure.
The environments <img src="https://render.githubusercontent.com/render/math?math=\chi"> and <img src="https://render.githubusercontent.com/render/math?math=\chi_0"> are compared using the kernel,
 
<img src="https://render.githubusercontent.com/render/math?math=k_{\chi_0}(\chi)= \int d\mathbf{r} \rho_{\chi}(\mathbf{r}) \rho_{\chi_0}(\mathbf{r}).">

where <img src="https://render.githubusercontent.com/render/math?math=\rho_{\chi}(\mathbf{r})"> is the atomic density around environment <img src="https://render.githubusercontent.com/render/math?math=\chi">.
In this way we obtain one value of the kernel per atom in the system.
We will then use as collective variable the number of <img src="https://render.githubusercontent.com/render/math?math=k_{\chi_0}(\chi)"> that are larger than some threshold.
This is equivalent to counting the number of atoms that have a crystalline environment.
We will also calculate the average of the <img src="https://render.githubusercontent.com/render/math?math=k_{\chi_0}(\chi)">.

You can find more details about the CV [in this article](https://aip.scitation.org/doi/abs/10.1063/1.5102104).

### Reweighting

The basic idea of reweighting is to calculate the unbiased ensemble average of an observable from a biased simulations.
Consider an observable $O(\mathbf{R})$ where $\mathbf{R}$ are the atomic coordinates.
The unbiased ensemble average is connected to the biased ensemble average via the formula:

$\langle O(\mathbf{R}) \rangle = \frac{\langle O(\mathbf{R}) e^{\beta V} \rangle_B}{\langle e^{\beta V} \rangle_B}$

where $\langle \cdot \rangle$ are unbiased averages and $\langle \cdot \rangle_B$ are biased averages. $\beta$ is the inverse temperature and $V$ is the bias potential.
In the case of metadynamics, the bias potential is not stationary and one needs to calculate a bias potential suitable for reweighting (see details [here](https://www.annualreviews.org/doi/abs/10.1146/annurev-physchem-040215-112229)).

### Free energy differences

A usual goal of enhanced sampling methods is to calculate differences in chemical potential between two (or more) states.
There are several methods to do this:
1. $\Delta F = F(s_A) - F(s_B)$ where $s_A$ is the position of the free energy minimum in basin A and $s_B$ is the position of the free energy minimum in basin B. This is an approximation, but a relatively good one if the difference in free energy is large and the barrier is large with respect to $k_B T$
2. A rigurous definition for the free energy difference is:

<p float="left">
  <img src="https://github.com/CSIprinceton/workshop-july-2023/raw/main/hands-on-sessions/day-4/9-metadynamics/eq1.png" width="150"> 
</p>

where $s^*$ is a watershed between the liquid and the solid.
3. An equivalent approach to the integration above, that does not require calculating $F(s)$ is:

<p float="left">
  <img src="https://github.com/CSIprinceton/workshop-july-2023/raw/main/hands-on-sessions/day-4/9-metadynamics/eq2.png" width="150"> 
</p>

where $\langle \cdot \rangle$ is an unbiased average, which can be calculated with reweighting, and $H(s-s*)$ is a unit step function at the watershed $s^*$

In the tutorial we will compare the results of these methods.

## Running and analyzing a metadynamics simulation

The folder ```metad-1350K``` contains the input files to run the simulation.
```metad-1350K/input.lmp``` is the LAMMPS input file, not to different from other input files except for the following line:

```
fix             1 all plumed plumedfile plumed.dat outfile log.plumed
```
which instructucts LAMMPS to use PLUMED with the input file ```plumed.dat``` and output file ```log.plumed```.
```metad-1350K/plumed.dat``` is the PLUMED input file and has three sections.
First, the definition of CV:

```
ENVIRONMENTSIMILARITY ...
 SPECIES=1-216
 SIGMA=0.04
 LATTICE_CONSTANTS=0.549 # in nm
 CRYSTAL_STRUCTURE=DIAMOND
 LABEL=es
 MORE_THAN={CUBIC D_0=0.05 D_MAX=0.95}
 MEAN
... 
```

targeting the cubic diamond structure with lattice constant 0.549 nm.
Instructions on how to determine SIGMA can be found [here](https://github.com/PabloPiaggi/masterclass-22-12/blob/main/ExerciseSolutions.ipynb).
The MORE_THAN keyword defines a CV ```es.morethan``` which counts the number of atoms with solid-like environments.
More information can be found in the [manual](https://www.plumed.org/doc-v2.9/user-doc/html/_e_n_v_i_r_o_n_m_e_n_t_s_i_m_i_l_a_r_i_t_y.html).

Second, the input for metadynamics:

```
METAD ...
 ARG=es.morethan
 SIGMA=1.5
 HEIGHT=60 # in kJ/mol
 PACE=500 # Every 500 steps is standard
 BIASFACTOR=150 # A barrier of 150 kT will be reduced to 1 kT once the bias is converged
 TEMP=1350 # Temperature in K
 LABEL=metad
 STRIDE=4 # Multistepping
 GRID_MIN=0
 GRID_MAX=216
 GRID_BIN=1000
 CALC_RCT
... 
```
We suggest that you explore the [manual](https://www.plumed.org/doc-v2.9/user-doc/html/_m_e_t_a_d.html) to understand each keyword.

Last, the instructions for printing output to a file named ```COLVAR``` every 1000 MD steps:

```
PRINT STRIDE=1000  ARG=* FILE=COLVAR
```

You can run the example with the command:
```
cd metad-1350K
lmp < input.lmp > out.lmp &
```
This will run a 5 ns long metadynamics simulation at 1700 K and 1 bar.
Several output files will be created.
```out.lmp``` file contains LAMMPS' output.
PLUMED's output files are ```log.plumed```, ```COLVAR```, and ```HILLS```.
Inspect these files.
```COLVAR``` will contain the collective variable, the bias potential, and other interesting quantities as a function of simulation time.
While the simulation is running you can check its progress by plotting the collective variable as a function of time using gnuplot.
The time is column 1 in ```COLVAR``` and the collective variable is column 3 in ```COLVAR```.
The ```HILLS``` file contains information about the Gaussians that make the potential.

Furthermore, you can calculate the FES with the command
```
plumed sum_hills --hills HILLS --mintozero
```
This will create a new file ```fes.dat```.
Plot the contents of this file (column 1 vs column 2) and track the convergence of the bias potential.

Further analysis is performed in the Jupyter Notebook ```Analysis.ipynb```.
You can open this notebook remotely following the instruction [here](https://github.com/CSIprinceton/workshop-july-2023/tree/main/hands-on-sessions/day-2/6-error-analysis#running-jupyter-notebook).
Inside this notebook you will see how to:
- plot the free energy surface (FES),
- assess convergence,
- perform reweighting,
- see the connection between biased and unbiased probability distributions,
- calculate free energy (and chemical potential) differences between the liquid and the solid.

Once you have done this for the temperature 1350 K.
We suggest that you run simulations at other temperatures close to 1350 K, for instance, 1300 K, 1400 K, etc.
Analyze again the FES and calculate free energy differences at these temperatures.
Some example scripts are provided in the Jupyter Notebook ```Analysis.ipynb```.

## Questions
- How does the FES change with temperature?
- How does the temperature affect the stability of the liquid and the solid?
- Can you calculate chemical potentials as a function of temperature?
- Can you find the coexistence temperature?
