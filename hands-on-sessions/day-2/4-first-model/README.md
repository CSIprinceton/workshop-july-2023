# Deep Modeling for Molecular Simulation
Hands-on sessions - Day 2 - July 12, 2023

Train your first first-principles machine-learning force field

## Aims

Using the DFT energies and forces obtained in the previous tutorial, train a model for the potential energy surface using DeePMD-kit.

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

## Training data

We will use the training data for silicon prepared on [hand-on session 3](https://github.com/CSIprinceton/workshop-july-2023/tree/main/hands-on-sessions/day-1/3-preparing-training-data).
These data consist of:
- Around 400 configurations of silicon in the cubic diamond crystal structure obtained using random displacements from equilibrium atomic positions. 
- Around 300 configurations of liquid silicon at 1700 K obtained in molecular dynamics simulations driven by the [Stillinger-Weber potential](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.31.5262).

Energies and forces for these configurations were obtained using DFT with the PBE functional.
You are encouraged to use the results of your own calculations.
However, you may also use the Quantum Espresso output files that we provide in the folders ```$TUTORIAL_PATH/hands-on-sessions/day-2/4-first-model/liquid-si-64``` and ```$TUTORIAL_PATH/hands-on-sessions/day-2/4-first-model/perturbations-si-64```.

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
```
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
```
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
```
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

Once the training is complete, we can proceed to **freeze** the model using ```dp freeze```.
This will create a deep potential file ```frozen_model.pb``` that can be used for inference (running MD or simply computing energies/forces).
It is useful to **compress** the model using ```dp compress -t input.json -i frozen_model.pb -o frozen_model_compressed.pb```.
This will create a model ```frozen_model_compressed.pb``` that can perform inference significantly faster than ```frozen_model.pb```.

