# Basics of 

Designed and written by Pablo Piaggi and Taehun Lee, Princeton University

Hands-on sessions - Day 1 - July 11, 2023

## Aims

This tutorial will demonstrate

## Objectives

This tutorial will cover the following:
- Basics for construction of training set
- Exercises:
  - Benchmarking DFT parameters
  - Geometry relaxation

## Backgrounds

Amongst the different methods for machine learning-based interatomic potentials model training, deep Potential (DP) is a typical method that fits interatomic potentials (potential energy surface, PES) by deep neural networks usually from datasets calculated by DFT-based methods. In the reference dataset preparation process, one also has to consider the expected accuracy of the final model or at what QM level one should label the data. In this tutorial, DFT PBE xc was used to calculate the **potential energy** and **atomic forces**. Several parameters and the control parameters of training the Deep Potential (DP) will be covered in the session.

Although described differently in various literature, the general process of training and utilizing machine learning interatomic potentials involves (1) exploration and labeling: constructing a database consisting of DFT calculation results for training, (2) training and validating models using the deepMD-kit, and (3) performing molecular dynamics simulations using the models.

<p float="left">
  <img src="https://github.com/CSIprinceton/workshop-july-2023/blob/6ed432411c4285a8dea9a77ce027c485d3e09b71/hands-on-sessions/day-1/2-quantum-espresso/kpoint.png" width="400"> 
</p>

Note: The left-hand side is taken from ref. 1. and the right-hand side of the figure is drawn in the style of ref. 2 and 3.

In particular for step (1), the construction of the dataset for training can be obtained in different ways, depending on the purpose of the final DP potential. To sample the local configurational space, which includes spatial configurations and chemical configurations, there are different ways to do this. Here, the choice can vary based on the purpose and aims of the DP. Before the training process, we need to prepare a dataset and convert them into the input form of DeepModeling.  Developing and utilizing machine learning potentials becomes more challenging when the element composition of the system becomes more diverse or when the system includes interfaces such as vacuum/solid and electrolyte/solid. It requires additional algorithms for predicting (meta)stable structures using global optimization or enhanced sampling methods. Further examples can be found for different systems, such as gas-phase and solid solutions.

In this tutorial, two representative approaches for exploration and labeling are introduced:
- Manual construction using random perturbation and vibrations of atoms from their equilibrium positions.
- MD simulations are effective methods for sampling local regions in the configuration space

## Exercises

### 1. Random perturbation
**exploration**: We will apply the random displacement in the atomic position of bulk Si supercell and vary the lattice constant.

First, we will read the structurally optimized structure of conventional bulk Si with 8 atoms. Then the suprecell will be constructued by (3 x 3 x 3). For each frame, the cell is perturbed by 5% and the atom positions are perturbed by 0.6 Angstrom. atom_pert_style indicates that the perturbation to the atom positions is subject to normal distribution. Please navigate to the `kpoints` directory where you will find a Python script named `kp.sh`.

```python
conf=ase.io.read('../pw-si-relaxed.out',format='espresso-out')
```

Parameters: Two parameters, max_displacement and max_cell_change, are defined to control the maximum displacement of atoms and the maximum fractional change in the cell, respectively. Random Perturbations: A loop is set up to perform a series of perturbations on the positions and lattice of the Si structure. Within each iteration, the initial positions and cell are copied. Random displacements within the defined maximum displacement are added to the positions, and random fractional changes within the defined maximum cell change are applied to the cell. The modified positions and cell are set back to the conf object using conf.set_positions() and conf.set_cell(), respectively.

```python
initial_positions=conf.get_positions()
initial_cell=conf.get_cell()

###############################################
# Random perturbations of positions and lattice
###############################################
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
    ase.io.write('pw-si-' + str(i) + '.in',conf, format='espresso-in',input_data=input_qe, pseudopotentials=pseudopotentials)
```

Then, it will generate multiple input files with random perturbations, allowing for sampling different configurations for subsequent QE calculations.
```
for i in `seq 0 99`
do
	srun --mpi=pmi2 \
	singularity run --nv \
     	/scratch/gpfs/ppiaggi/Simulations/QuantumEspressoGPU/quantum_espresso_qe-7.1.sif \
     	pw.x -input pw-si-$i.in -npool 2 > pw-si-$i.out
done
```

**labeling:** Now that we selected a set of atomic configurations from the exploration step, we should label these configuration. Labeling consists of evaluation of atomic forces and energy using first-principles methods. We use the PBE functional, as implemented in the PW code of Quantum-ESPRESSO, to evaluate first-principles energy and forces. At the end of this step, raw files will be extracted from PW outputs and later appended to the existing DP training data. Here we only introduced the required files full list can be found here. You can parse the **spatial configurations**, **potential energy**, and **atomic forces** using ASE-calculator. 

<div align="center">
	
ID       | Property                | Raw file     | Shape                  
-------- | ----------------------  | ------------ | -----------------------
type     | Atom type indexes       | type.raw     | Natoms                 
coord    | Atomic coordinates      | coord.raw    | Nframes \* Natoms \* 3  in Å
box      | Boxes                   | box.raw      | Nframes \* 3 \* 3       in Å
energy   | Frame energies          | energy.raw   | Nframes                 in eV
force    | Atomic forces           | force.raw    | Nframes \* Natoms \* 3  in eV/
virial   | Frame virial            | virial.raw   | Nframes \* 9 in eV       

</div>

Note: The left-hand side is taken from [here](https://github.com/deepmodeling/deepmd-kit/blob/master/doc/data/system.md). Box and virial: in the order `XX XY XZ YX YY YZ ZX ZY ZZ`.

```python
import numpy as np
import ase.io
from ase.calculators.espresso import Espresso

#################################
# Coordinates
#################################
file_coord = open("coord.raw", "w")
file_energy = open("energy.raw", "w")
file_force = open("force.raw", "w")
file_virial = open("virial.raw", "w")
file_box = open("box.raw", "w")
file_type = open("type.raw", "w")
types_written=False
for i in range(101):
    try:
        conf=ase.io.read('pw-si-' + str(i) + '.out',format='espresso-out')
    except:
        print("Configuration " + str(i) + " could not be read")
    else:
        try:
            conf.get_forces()
        except:
            print("Forces missing from file" + str(i))
        else:
            file_coord.write(' '.join(conf.get_positions().flatten().astype('str').tolist()) + '\n')
            file_energy.write(str(conf.get_potential_energy()) + '\n')
            file_force.write(' '.join(conf.get_forces().flatten().astype('str').tolist()) + '\n')
            file_virial.write(' '.join(conf.get_stress(voigt=False).flatten().astype('str').tolist()) + '\n')
            file_box.write(' '.join(conf.get_cell().flatten().astype('str').tolist()) + '\n')
            if (not(types_written)):
                types = np.array(conf.get_chemical_symbols())
                types[types=="Si"]="0"
                file_type.write(' '.join(types.tolist()) + '\n')
                types_written=True
file_coord.close()
file_energy.close()
file_force.close()
file_virial.close()
file_box.close()
file_type.close()
```

### 2. MD simulations
For example, vibrations of atoms around their equilibrium positions in solids (MD), and changing the local environment using MD LAMMPS snapshots. Amorphous (liquid) structures with different densities were considered, which were obtained by annealing the above structures at high temperatures and different pressures.

## Additional considerations and references
- Parhsing dpdata python module
- this process can be automatically done with dpGen
- [1]:
- [2]:
- [3]:
