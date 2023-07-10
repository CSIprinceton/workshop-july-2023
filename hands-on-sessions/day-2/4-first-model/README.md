# Deep Modeling for Molecular Simulation
Hands-on sessions - Day 2 - July 12, 2023

Train your first first-principles machine-learning force field

## Aims

Using the DFT energies and forces obtained in the previous tutorial, train a model for the potential energy surface (PES) using DeePMD-kit.

## Objectives

The objectives of this tutorial session are:
- Train a DeePMD model for the potential energy surface of silicon
- Prepare DFT output data for the training process
- Become familiar with the inputs and outputs of the training process
- Run molecular dynamics simulations driven by the DeePMD model
- Identify strengths and limitations of a rudimentary model
- Use an ensemble of models to estimate the errors in the forces

## Prerequisites

It is assumed that the student has completed all hands-on sessions from [day 1](https://github.com/CSIprinceton/workshop-july-2023/tree/main/hands-on-sessions/day-1) of this workshop.

## Theory

### Model

In the DeePMD model that we will train for the PES of silicon, the total energy $E$ of a configuration of $N$ atoms with atomic coordinates $\mathbf{R}$ is written as a sum over per-atom energies $E_i$, i.e,

$E(\mathbf{R})=\sum\limits_{i=1}^N E_i = \sum\limits_{i=1}^N E^{\alpha_i}(\mathbf{R}_i)$

where $\mathbf{R}_i$ are the relative atomic coordinates of $N_i$ neighbors in an environment with cutoff $r_c$ around atom $i$, $\alpha_i$ is the atom type of atom $i$, and $E^{\alpha_i}$ is an energy function for atoms of the chemical species $\alpha_i$.
In order to preserve the natural symmetries of the problem, i.e., rotation and permutation of atoms of the same type, we define a vector of descriptors $\mathbf{D}_i$ for atom $i$.
Then, the energy of a configuration can be written as,

$E(\mathbf{R})=\sum\limits_{i=1}^N E^{\alpha_i}(\mathbf{D}_i)$

The starting point for the definition of the descriptors $\mathbf{D}_i$ is a continuous and differentiable switching function,

<p float="left">
  <img src="https://github.com/CSIprinceton/workshop-july-2023/raw/main/hands-on-sessions/day-2/4-first-model/eq1.png" width="350">
</p>

where $u=(r - r_s)/(r_c - r_s)$, and $r_s$ and $r_c$ are smooth and hard cutoffs, respectively.
Next, we construct a matrix $\mathbf{R}_i \in \mathbb{R}^{N_i \times 4}$ of generalized coordinates with rows,

<p float="left">
  <img src="https://github.com/CSIprinceton/workshop-july-2023/raw/main/hands-on-sessions/day-2/4-first-model/eq2.png" width="350">
</p>

where $(x_{ij},y_{ij},z_{ij})$ is the distance vector from atom $j$ to atom $i$, and $r_{ij}$ is the norm of such distance.
Furthermore, we define an embedding matrix $\mathbf{G}^i \in \mathbb{R}^{N_i \times M_1}$ with row $j$ given by,  

<p float="left">
  <img src="https://github.com/CSIprinceton/workshop-july-2023/raw/main/hands-on-sessions/day-2/4-first-model/eq3.png" width="200">
</p>


where $g^{\alpha_i,\alpha_j}$ is a function that maps a scalar into $M_1$ outputs, and is different for each pair of chemical species $\alpha_i$ and $\alpha_j$.
We also define a secondary embedding matrix $\mathbf{G}'^i\in\mathbb{R}^{N_i\times M_2}$ with the first $M_2< M_1$ columns of $\mathbf{G}^i$.

With these ingredients, we now write the descriptor matrix $\mathbf{D}_i \in \mathbb{R}^{M_1 \times M_2}$ as,

<p float="left">
  <img src="https://github.com/CSIprinceton/workshop-july-2023/raw/main/hands-on-sessions/day-2/4-first-model/eq4.png" width="200">
</p>

which is subsequently flatten into a vector of $M_1 \times M_2$ elements and is used as input in the equation above.
In our simulations, we will use a model for a single species, namely, Si.
$E^{\alpha_i}$ will be represented by a neural network with three layers and 80 neurons per layer, and $g^{\alpha_i,\alpha_j}$ will be represented by a three-layer neural network with sizes 20, 40 and 80, respectively.
Other parameters of our model are $M_1=80$, $M_2=16$, $r_s=3$ Angstrom, and $r_c=6$ Angstrom.

### Loss function

The parameters in the neural networks $E^{\alpha_i}$ and $g^{\alpha_i,\alpha_j}$ described above are determined through the minimization of the following loss function,
    
$\mathcal{L} = \frac{1}{N_\mathcal{B}} \left (\sum_{l \in \mathcal{B}}  \frac{w_{\epsilon}}{N_l} \left | E_l- E(\mathbf{R}^l)\right |^2  + \frac{w_{f}}{3N_l}  \left \| \mathbf{F}_l- \mathbf{F}(\mathbf{R}^l) \right \|^2  \right)$

where $\mathcal{B}$ is a mini-batch (i.e., a subset of the training set) with $N_\mathcal{B}$ atomic configurations,  $w_{\epsilon}$ and $w_{f}$ are weights. 
Furthermore, $E_l$ and $F_l$ are reference energies and forces, $E(\mathbf{R}^l)$ and $\mathbf{F}(\mathbf{R}^l)=-\boldsymbol\nabla_\mathbf{R} E(\mathbf{R}^l)$ are the energy and force predictions of our model for configuration $l$ in the minibatch, and $\mathbf{R}^l$ and $N_l$ are the atomic coordinates and the number of atoms in configuration $l$.

### Optimizer

We will train the models using the Adam optimizer with learning rate $\alpha(i)=\alpha_0 \lambda^{i/\tau}$ where $\alpha_0=0.002$ is the initial learning rate, $\lambda=0.97$, $\tau=5\times10^3$, and $i$ is the step number.
The batch size $N_\mathcal{B}$ is set to one and we will train for a total number of steps equal to $2\times 10^5$.
$w_{\epsilon}$ and $w_{f}$ were varied according to $w_{\epsilon}(i)=w_{\epsilon}^1+(w_{\epsilon}^0-w_{\epsilon}^1)\lambda^{i/\tau}$ and $w_f(i)=w_f^1+(w_f^0-w_f^1)\lambda^{i/\tau}$, with $w_{\epsilon}^0=0.02$, $w_{\epsilon}^1=1$, $w_f^0=1000$, and $w_f^1=1$.
This scheme gives a higher weight to the force term in the loss function at the beginning of the training process, and by the end of it both the energy and force term have equal weights.

## Training data

We will use the training data for silicon prepared on [hand-on session 3](https://github.com/CSIprinceton/workshop-july-2023/tree/main/hands-on-sessions/day-1/3-preparing-training-data).
These data consist of:
- Around 400 configurations of silicon in the cubic diamond crystal structure obtained using random displacements from equilibrium atomic positions. 
- Around 300 configurations of liquid silicon at 1700 K obtained in molecular dynamics simulations driven by the [Stillinger-Weber potential](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.31.5262).

Energies and forces for these configurations were obtained using DFT with the PBE functional.
You are encouraged to use the results of your own calculations.
However, you may also use the Quantum Espresso output files that we provide in the folders ```$TUTORIAL_PATH/hands-on-sessions/day-2/4-first-model/example-data/liquid-si-64``` and ```$TUTORIAL_PATH/hands-on-sessions/day-2/4-first-model/example-data/perturbations-si-64```.

First, we have to extract the energies and forces from the Quantum Espresso output files and organize them in the .raw filetype suitable for DeePMD.
There are many ways to carry out this task.
Here, we propose to use a script ```get_raw.py``` based on [ASE](https://wiki.fysik.dtu.dk/ase/) that we provide in the folder ```$TUTORIAL_PATH/hands-on-sessions/day-2/4-first-model/scripts/```.
You can execute this script in the folders containing the Quantum Espresso output files to obtain the following files:
- ```energy.raw```
- ```force.raw```
- ```coord.raw```
- ```box.raw```
- ```type.raw```

See the [manual](https://docs.deepmodeling.com/projects/deepmd/en/master/data/data-conv.html#raw-format-and-data-conversion) for an explanation of the format and units of these files.
The last step is to use the ```raw_to_set.sh``` utility in DeePMD to have the data ready for the training process.
You can execute this utility in each folder containing .raw data files using the command:

```$PATH_TO_DEEPMD_KIT/data/raw/./raw_to_set.sh 101```

The data should now be ready for the training process!
Another excellent way to convert output of electronic-structure calculation into the DeePMD-kit format is using [dpdata](https://docs.deepmodeling.com/projects/deepmd/en/master/data/dpdata.html).

We shall see below whether a DeePMD model trained on the configurations described above is able to drive the dynamics of this system, while preserving a high-accuracy.
Can you guess if the model will be good for the liquid, the solid, none, or both? Why?
Let's make a poll in the classroom!

## Training process

An example input script for the training process is provided in ```scripts/input.json```.
Before executing it, let's analyze its contents.
The first block is the model definition:
```json
   "model": {
        "type_map":     ["Si"],
        "descriptor": {
            "type": "se_a",
            "sel": [30],
            "rcut_smth": 3.0,
            "rcut": 6.0,
            "neuron": [
                20,
                40,
                80
            ],
            "axis_neuron": 16,
            "seed": 25875,
        },
        "fitting_net": {
            "neuron": [
                80,
                80,
                80
            ],
            "resnet_dt": true,
            "seed": 25875,
        },
```
We will discuss this input in the classroom and you can also find further information [here](https://docs.deepmodeling.com/projects/deepmd/en/master/model/train-se-e2-a.html).

The next two blocks specify options for the optimization process and the definition of the loss function:
```json
    "learning_rate": {
        "start_lr": 0.002,
        "decay_steps": 500,
    },
    "loss": {
        "start_pref_e": 0.02,
        "limit_pref_e": 1,
        "start_pref_f": 1000,
        "limit_pref_f": 1,
        "start_pref_v": 0,
        "limit_pref_v": 0,
    },
```

In the last block we will specify, among other things, the training and validation data:
```json
    "training": {
        "stop_batch": 200000,
        "disp_file": "lcurve.out",
        "disp_freq": 2000,
        "save_freq": 20000,
        "save_ckpt": "model.ckpt",
        "validation_data": {
            "systems": [
                "<SOME_FOLDER>/perturbations-si-64/0.05A-2p"
            ],
            "batch_size":       "auto"
        },
        "training_data": {
            "systems": [
                "<SOME_FOLDER>/perturbations-si-64/0.01A-1p",
                "<SOME_FOLDER>/perturbations-si-64/0.1A-3p",
                "<SOME_FOLDER>/perturbations-si-64/0.2A-5p",
                "<SOME_FOLDER>/liquid-si-64/trajectory-lammps-1700K-1bar/extracted-confs",
                "<SOME_FOLDER>/liquid-si-64/trajectory-lammps-1700K-10000bar/extracted-confs",
                "<SOME_FOLDER>/liquid-si-64/trajectory-lammps-1700K-neg10000bar/extracted-confs"
                        ],
            "batch_size":       "auto"
        }
    }
```

Edit <SOME_FOLDER> to point to the directory with your training data.
Note that we have chosen a somewhat arbitrary separation between training and validation data.

Now it's time to start training the model for the potential energy surface!
Execute ```dp train input.json``` to start training.
The training process should take about 15 minutes.
You can monitor its progress in the file lcurve.out.
The first few lines are as follows:
```
#  step      rmse_val    rmse_trn    rmse_e_val  rmse_e_trn    rmse_f_val  rmse_f_trn         lr
      0      1.96e+01    6.35e+00      9.76e-01    9.36e-01      6.20e-01    1.98e-01    2.0e-03
   2000      2.64e+00    1.06e+00      6.86e-02    7.54e-02      8.84e-02    3.47e-02    1.8e-03
   4000      1.39e+00    7.93e+00      3.97e-02    4.74e-02      4.94e-02    2.83e-01    1.6e-03
   6000      2.19e+00    7.75e+00      4.27e-02    1.04e-01      8.28e-02    2.94e-01    1.4e-03
   8000      1.24e+00    7.51e+00      4.99e-02    8.79e-02      4.89e-02    3.02e-01    1.2e-03
  10000      1.06e+00    5.91e+00      6.94e-02    4.24e-02      4.23e-02    2.53e-01    1.1e-03
```
where the columns represent the training steps, the total RMS error (val-validation and trn-training), the RMS error in energy, the RMS error in the forces, and the learning rate.
You can plot the number of steps vs the RMS errors to follow the progress of the training process.

Once the training is complete, we can proceed to freeze the model using ```dp freeze```.
This will create a deep potential file ```frozen_model.pb``` that can be used for inference (running MD or simply computing energies/forces).
It is useful to [compress](https://pubs.acs.org/doi/full/10.1021/acs.jctc.2c00102) the model using ```dp compress -t input.json -i frozen_model.pb -o frozen_model_compressed.pb```.
This will create a model ```frozen_model_compressed.pb``` that can perform inference significantly faster than ```frozen_model.pb```.

## Running molecular dynamics simulations

Equipped with the model trained in the previous section, we will now run molecular dynamics simulations.
A LAMMPS script to simulate solid silicon in the cubic diamond structure can be found at ```molecular-dynamics/solid/input.lmp```.
This input file has been annotated to help you understand the purpose of each line.
The simulation uses a thermostat and barostat to mantain a temperature of 300 K and a pressure of 1 bar.

The lines of the input file that instruct the code to use the DeePMD model is:
```
pair_style      deepmd ../frozen_model_1_compressed.pb ../frozen_model_2_compressed.pb ../frozen_model_3_compressed.pb ../frozen_model_4_compressed.pb out_file md.out out_freq ${out_freq}
pair_coeff      * *
```
where ```frozen_model_?_compressed.pb``` are four models trained on the same data and different initial random seeds.
These four models are employed to estimate the errors in the forces.
We define the error $\epsilon_i$ in the $i$-th force component as $\epsilon_i^2 = \langle | f_i-\bar{f}_i |^2 \rangle$, where $\bar{f}_i = \langle f_i \rangle$ and the average $\langle \cdot \rangle$ is taken over the ensemble of models.
The average, minimum, and maximum errors in the forces are reported every ```out_freq``` steps in the file ```md.out```.
Four models are provided in ```molecular-dynamics/frozen_model_?_compressed.pb```, but you are encourage to use your own model trained in the previous section.
Also, you may want to share models with other participants.

You can now run the simulation using the command,
```
lmp < input.lmp
```
You can now copy the trajectory ```si.lammps-dump-text``` to your laptop and visualize it with Ovito.
Does it show the expected behavior for a solid?

Once that you have loaded the LAMMPS dump file into Ovito, you can color the atoms according to the degree of order around them.
Apply the ```Identify diamond structure``` modifier that can be chosen from the ```Add modification``` dropdown menu.
For reference, below we show liquid and solid configurations colored with the modifier ```Identify diamond structure```.

<p float="left">
  <img src="https://github.com/PabloPiaggi/Crystallization-of-Silicon/raw/master/si-liquid.png" width="250"> 
  <img src="https://github.com/PabloPiaggi/Crystallization-of-Silicon/raw/master/si-solid.png"  width="250">
</p>

You can also plot thermodynamic properties of the system that have been printed to the file ```thermo.txt```.

Next, let's analyze the contents of the file ```md.out```, which should be similar to:
```
#       step         max_devi_v         min_devi_v         avg_devi_v         max_devi_f         min_devi_f         avg_devi_f
           0       6.141403e-03       6.360340e-07       3.476040e-03       6.160679e-03       1.564173e-03       3.614160e-03
        1000       3.983807e-03       3.534347e-04       2.002133e-03       2.265209e-02       5.212460e-03       1.369438e-02
        2000       4.729147e-03       3.986826e-04       2.127513e-03       1.862923e-02       5.243879e-03       1.081513e-02
        3000       6.839780e-03       2.654651e-04       3.016102e-03       2.869552e-02       5.306404e-03       1.235991e-02
```
We suggest that you plot steps (column 1) vs the maximum deviation of the forces (column 5), and the steps (column 1) vs the average deviation of the forces (column 7).
Are the value of the errors stable? What are their magnitudes? Can you conclude that the model is well-trained to describe the solid, or does it require further training?

Now that we have studied the performance of our rudimentary model for the solid, let's run a simulation for liquid silicon.
An appropriate LAMMPS file is provided in ```molecular-dynamics/liquid/input.lmp```.
The simulation uses a thermostat and barostat to mantain a temperature of 1700 K (our guess for the melting temperature) and a pressure of 1 bar.
Once the simulation has completed, analyze it with the same steps described above for the solid.
Is the model suitable to describe liquid silicon? Why?

In [hands-on session 5](https://github.com/CSIprinceton/workshop-july-2023/tree/main/hands-on-sessions/day-2/5-active-learning) you will learn a technique to systematically improve the stability and accuracy of the models.


