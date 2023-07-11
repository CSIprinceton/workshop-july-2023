# Basics of Preparing Training Data

Designed and written by Pablo Piaggi and Taehun Lee, Princeton University

Hands-on sessions - Day 1 - July 11, 2023

## Aims

This tutorial offers an overview of the process of preparing initial training data for models for the ab initio potential energy surface based on the deep potential (DP or DeePMD) methodology. It focuses on two techniques: random perturbations and molecular dynamics simulations.

## Objectives

The objectives of this tutorial session are:
- Learn to prepare data to train a machine-learning model for the potential energy surface (PES)
- Create configurations suitable to train a model for a crystalline solid using random perturbations from the equilibrium atomic positions
- Create configurations suitable to train a model for a liquid using molecular dynamics driven by another force field
- Label the above configurations with their energy and forces using density-functional theory (DFT) calculations
- Perform a large number of DFT calculations using Quantum Espresso and the PBE functional
- Prepare files in a format appropriate for training a model for the PES using DeePMD-kit

## Prerequisites

It is assumed that the student has completed hands-on [session 2](https://github.com/CSIprinceton/workshop-july-2023/tree/main/hands-on-sessions/day-1/2-quantum-espresso) of this workshop.

## Background

DP models for the potential energy surface are based on deep neural networks, and are typically trained on datasets with configurations for which the potential energy and the forces have been calculated using quantum-mechanical density-functional theory (DFT). The process of training models involves the following steps: (1) exploration: constructing a dataset consisting of atomistic configurations by exploring the configuration space, (2) labeling: calculating energies and forces for the configurations via DFT calculations, and (3) training and validating the DP model using the [DeepMD-kit](https://docs.deepmodeling.com/projects/deepmd/en/master/). In this hands-on session we will cover the first two steps, while step (3), i.e. training, will be covered in [hands-on sessions 4](https://github.com/CSIprinceton/workshop-july-2023/tree/main/hands-on-sessions/day-2/4-first-model).

In particular, for step (1), the construction of the training dataset can be done in various ways, depending on the purpose of the final DP model. The database consists of multiple frames, with each frame containing chemical and configurational information, DFT-computed forces, and potential energy. It is important to note that developing and utilizing DP can be more challenging when the system has diverse element compositions or includes interfaces (or phase boundaries) such as vacuum/solid and water/solid. Exploring and labeling such complex systems may require additional algorithms or techniques, such as global optimization or enhanced sampling methods.

## Exercises
In this tutorial, we will walk you through the process of preparing a training dataset for `DeepMD-kit`. You will sample a set of bulk Si structures (liquid and solid) using two different techniques to create a diverse set of starting configurations: 
- For solid Si in the cubic diamond crystal structure: create random perturbations of atomic positions and cell vectors from their equilibrium values in the solid.
- For liquid Si: Molecular dynamics simulations of the liquid using the semiempirical Stillinger-Weber potential - [Stillinger and Weber, Phys. Rev. B, v. 31, p. 5262, (1985)](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.31.5262)

Using the sampled structures, you will perform DFT calculations with QE to obtain the potential energy and atomic forces for each structure (or frame). The calculated data, including the potential energy and atomic forces, will be converted into the appropriate input formats required by `DeepMD-kit` for subsequent analysis and training of interatomic potential models.

### Crystalline Si - Random perturbations

**1. Exploration**: Please navigate to the `perturbations-si-64/0.01A-1p` directory where you will find a Python script named `perturbations.py`. Using this script, first, the structurally optimized structure of bulk Si with 8 atoms will be read as ASE atoms object. Then the supercell will be constructed by expanding the unit cell using (2 x 2 x 2) transformation vector which yields the supercell with 64 atoms.

```python
from ase.build import make_supercell

bulk_si = ase.io.read('../pw-si-vc_relax.out',format='espresso-out')
P = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
conf = make_supercell(bulk_si, P)
```

Then, you will apply random displacements to the atomic positions of a bulk Si supercell and vary the lattice parameters using ASE atoms object. Random displacements within the defined maximum displacement will be added to the equilibrium atomic positions, and random fractional changes within the defined maximum cell change will be applied to the lattice parameters. The Python script will generate a total of `100 frames` and corresponding QE input files. In each frame, the cell and the atomic positions will be perturbed by a maximum of `1 %` and `0.01 Å`, respectively, from the ground state bulk Si structure. The degree of perturbation of the atomic positions and the cell for each frame follows a uniform distribution.

```python
initial_positions = supercell.get_positions()
initial_cell = supercell.get_cell()

max_displacement=0.01 # Maximum displacement in angstrom
max_cell_change=0.01  # Maximum fractional change in cell

num_iterations=100

for i in range(num_iterations):
    positions=np.copy(initial_positions)
    cell=np.copy(initial_cell)

    # Displace each coordinate randomly
    positions += np.random.rand(positions.shape[0],positions.shape[1])*2*max_displacement - max_displacement
    conf.set_positions(positions)

    # Scale each cell component randomly
    cell *= 1-(np.random.rand(cell.shape[0],cell.shape[1])*2*max_cell_change-max_cell_change)
    conf.set_cell(cell,scale_atoms=True)
	
    # Write QE input
    write('pw-si-' + str(i) + '.in',conf, format='espresso-in',input_data=input_qe, pseudopotentials=pseudopotentials)
```

Let's type `python perturbations.py` to generate QE input files. Let's play with the `max_displacement` and `max_cell_change` variables by constructing the different datasets to sample enough chemical spaces (refer to `0.05A-2p`, `0.1A-3p`, `0.2A-5p` directories).

<br/>


**2. Labeling:** Now that you have generated a set of atomic configurations from the exploration step, the next step is to label these configurations, i.e., calculate energies and forces using DFT. The following `job.sh` bash script executes Quantum Espresso on the 100 input files that we just created by performing SCF DFT calculation for each frame to evaluate the forces and energy:
```shell
conda deactivate
export PW=/home/deepmd23admin/Softwares/QuantumEspresso/q-e-qe-7.0/bin/pw.x
for i in `seq 0 99`
do
        mpirun -np 1 $PW -input pw-si-$i.in > pw-si-$i.out
done
```
To run these DFT tasks in the background, you can use
```
chmod 777 job.sh
nohup ./job.sh &
```
To monitor the processes, you can use
```
ps aux|grep job.sh
ps aux|grep pw.x
```
If you want to shutdown the calculation, execute `kill PROCESSID` where `PROCESSID` is the id of the process `job.sh`.

For each input file `pw-si-$i.in`, Quantum Espresso will create a `pw-si-$i.out` file which contains the potential energy, the forces, and other useful information. 

We have to extract the raw data from the PW outputs and convert them into the input format required by `deepMD-kit` for training. A full list of these files can be found [here](https://github.com/deepmodeling/deepmd-kit/blob/master/doc/data/system.md). The following is a description of the basic `deepMD-kit` input formats:

<br/>

<div align="center">
	
ID       | Property                | Raw file     | Shape                  
-------- | ----------------------  | ------------ | -----------------------
type     | Atom type indexes       | type.raw     | Natoms                 
coord    | Atomic coordinates      | coord.raw    | Nframes \* Natoms \* 3  in Å
box      | Boxes                   | box.raw      | Nframes \* 3 \* 3       in Å
energy   | Frame energies          | energy.raw   | Nframes                 in eV
force    | Atomic forces           | force.raw    | Nframes \* Natoms \* 3  in eV/Å
virial   | Frame virial            | virial.raw   | Nframes \* 9 in eV       

<em>The table is taken from [here](https://github.com/deepmodeling/deepmd-kit/blob/master/doc/data/system.md). `Box` and `virial`: in the order `XX XY XZ YX YY YZ ZX ZY ZZ`.</em>
</div>

<br/>

You can parse the atomic structures, potential energy, and atomic forces from QE outputs using the ASE calculator and a numpy-based python script named `get_raw.py`.

```python
import numpy as np
import ase.io
from ase.calculators.espresso import Espresso

# Open output files for writing
file_coord = open("coord.raw", "w")     # Coordinates
file_energy = open("energy.raw", "w")   # Potential energy
file_force = open("force.raw", "w")     # Forces
file_virial = open("virial.raw", "w")   # Virial stress
file_box = open("box.raw", "w")         # Cell dimensions
file_type = open("type.raw", "w")       # Atom types

types_written = False

for i in range(100):
    try:
        conf = ase.io.read('pw-si-' + str(i) + '.out', format='espresso-out')
    except:
        print("Configuration " + str(i) + " could not be read")
    else:
        try:
            conf.get_forces()
        except:
            print("Forces missing from file" + str(i))
        else:
            # Write data to respective output files
            file_coord.write(' '.join(conf.get_positions().flatten().astype('str').tolist()) + '\n')
            file_energy.write(str(conf.get_potential_energy()) + '\n')
            file_force.write(' '.join(conf.get_forces().flatten().astype('str').tolist()) + '\n')
            file_virial.write(' '.join(conf.get_stress(voigt=False).flatten().astype('str').tolist()) + '\n')
            file_box.write(' '.join(conf.get_cell().flatten().astype('str').tolist()) + '\n')
            
            if not types_written:
                types = np.array(conf.get_chemical_symbols())
                types[types == "Si"] = "0"
                file_type.write(' '.join(types.tolist()) + '\n')
                types_written = True

# Close output files
file_coord.close()
file_energy.close()
file_force.close()
file_virial.close()
file_box.close()
file_type.close()
```
Execute this script by typing 
```
python get_raw.py
```

Now let's verify if this script successfully generates the files `coord.raw`, `energy.raw`, `force.raw`, `virial.raw`, `box.raw`, and `type.raw`. It's important to note that while the raw format is not directly supported for training, NumPy and HDF5 binary formats are supported. 

To convert the prepared raw files to the NumPy, you can utilize the tool provided in the DeePMD-kit `raw_to_set.sh` by

```
/home/deepmd23admin/Softwares/deepmd-kit/data/raw/raw_to_set.sh
```

### Liquid Si - MD simulations with another force field

**1. Exploration**: We will now run molecular dynamics simulation of liquid Si with the Stillinger-Weber force field using LAMMPS.
The LAMMPS input files can be found in the directory `liquid-si-64/trajectory-lammps-1700K-1bar` for a simulation at 1 bar and 1700 K (approximate melting temperature of Stillinger-Weber Si).
The MD simulations can be run with the command,

```shell
lmp < start.lmp
```
and the simulation takes a couple of minutes to complete.
The atomic coordinates are written every 10 ps to the file `si.lammps-dump-text` in LAMMPS dump format.

> **Note** Element infomation can be saved to LAMMPS dump file if the followed commands are used. The `xs ys zs` are scaled coordinates. Other properties can be save as well, namely, atom velocities `vx vy vz`. (See more details in [doc](https://docs.lammps.org/dump.html).) When a dump file with element info is visualised by OVITO, particles will have corresponding radii and colours.
```
dump                    myDump all custom ${out_freq2} si.lammps-dump-text id type element xs ys zs
dump_modify             myDump element Si
```

**2. Labeling:** We can now extract configurations from this trajectory and create input files to perform DFT calculations with the python script `get_configurations.py` which reads:
```python
import numpy as np
import ase.io
from ase.calculators.espresso import Espresso
import os

################################
# QE options
################################

pseudopotentials = {'Si': 'Si_ONCV_PBE-1.0.upf'}

input_qe = {
            'calculation':'scf',
            'outdir': './',             
            'pseudo_dir': './',         
            'tprnfor': True,        
            'tstress': True,        
            'disk_io':'none',
            'system':{
              'ecutwfc': 30,
              'input_dft': 'PBE',
             },
            'electrons':{
               'mixing_beta': 0.5,
               'electron_maxstep':1000,
             },
}

os.system('mkdir extracted-confs')

# Load trajectory
traj=ase.io.read('si.lammps-dump-text',format='lammps-dump-text',index=':')
step=1
counter1=0 # Number of configurations written
counter2=0 # Frame number
for conf in traj:
   if ((counter2%step)==0):
      species=np.array(conf.get_chemical_symbols())
      species=np.full(shape=species.shape,fill_value="Si")
      conf.set_chemical_symbols(species)
      ase.io.write('extracted-confs/pw-si-' + str(counter1) + '.in',conf, format='espresso-in',input_data=input_qe, pseudopotentials=pseudopotentials)
      counter1 += 1
   counter2 += 1
```
Execute `python get_configurations.py`.
This will create a folder `extracted-confs` with 100 Quantum Espresso input files with the atomic configurations extracted from the trajectory `si.lammps-dump.text`.
You can now perform DFT calculations on all of these configurations using the same script `job.sh` as above.
Next, extract the raw data files with the script `get_raw.py` as above, and convert them into the appropriate format for `DeePMD-kit` using the `raw_to_set.sh` script.

The above calculations can be repeated for two other pressures, namely, +- 10 kbar, in order to sample a broad range of volumes.
This is illustrated in the folders `trajectory-lammps-1700K-10000bar` and `trajectory-lammps-1700K-neg10000bar`.

> **Note** ASE can get the correct `chemical_symbols` if the LAMMPS dump file has `element` info. Otherwise, use `traj=ase.io.read('si.lammps-dump-text',format='lammps-dump-text',index=':',specorder=[“Si”])` to pass the chemical symbols. If there are two types in LAMMPS, namely, Si and O, then `specorder=["Si", "O"]`.

## Outcome

By the end of this hands-on session you should have around 400 configurations for solid Si and 300 configurations for liquid Si ready to be used to train a machine-learning force field with `DeePMD-kit`.

## Additional considerations and references
- [DP-GEN](): A user-friendly and **automatic** Python software for generating accurate and efficient DP models dependent on `DeePMD-kit`.
- [Dpdata](https://github.com/deepmodeling/dpdata): A python package for manipulating data formats of software, including DeePMD-kit, QE, VASP, LAMMPS, GROMACS, Gaussian, and CP2K.
