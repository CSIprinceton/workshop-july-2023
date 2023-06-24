# Basics of 

Designed and written by Pablo Piaggi and Taehun Lee, Princeton University

Hands-on sessions - Day 1 - July 11, 2023

## Aims and Objectives

This tutorial will demonstrate

This tutorial will cover the following:
- Backgrounds for construction of training set
- Exercises:
  - Benchmarking DFT parameters
  - Geometry relaxation

## Backgrounds

As one branch of machine learning-based interatomic potential model training, Deep Potential (DP) fits interatomic potentials (potential energy surface, PES) using deep neural networks, typically from datasets calculated by DFT. The process of training DP involves the following steps: (1) exploration and labeling: constructing a database consisting of DFT calculations, (2) training and validating DP models using the [deepMD-kit], and (3) performing molecular dynamics simulations using the DP model interfaced with [LAMMPS].

<p float="left">
  <img src="[https://github.com/CSIprinceton/workshop-july-2023/blob/6ed432411c4285a8dea9a77ce027c485d3e09b71/hands-on-sessions/day-1/3-preparing-training-data/protocol_sampling.png](https://github.com/CSIprinceton/workshop-july-2023/blob/cf3e79c9402f39423c75a61cbed22e4a14dc6313/hands-on-sessions/day-1/3-preparing-training-data/protocol_sampling.png)" width="400"> 
</p>

Note: The left-hand side is taken from ref. 1. and the right-hand side of the figure is drawn in the style of ref. 2 and 3.

In particular, for step (1), the construction of the training dataset can be done in various ways, depending on the purpose of the final DP potential. The database consists of multiple frames, with each frame containing chemical and configurational information, DFT-computed forces, and potential energy. It is important to note that developing and utilizing DP can be more challenging when the system has diverse element compositions or includes interfaces such as vacuum/solid and water/solid. Exploring and labeling such complex systems may require additional algorithms or techniques, such as global optimization or enhanced sampling methods (shown in the above figure).

In this tutorial, two simple and representative approaches for exploration and labeling, are introduced:
- Manual construction using random perturbation and vibrations of atoms and cells from their equilibrium positions.
- Molecular dynamics simulations.

## Exercises
In this tutorial, DFT PBE xc was used to calculate the **potential energy** and **atomic forces**. bulk Si, based on DFT calculations with QE

### 1. Random perturbation
**exploration**: First, the structurally optimized structure of conventional bulk Si with 8 atoms will be read as ASE atoms object. Then the supercell will be constructed by expanding the unit cell by transformation (3 x 3 x 3) which yields the supercell with 64 atoms which is invoked as follows: 

```python
from ase.build import make_supercell

bulk_si = ase.io.read('../pw-si-relaxed.out',format='espresso-out')
P = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
supercell = make_supercell(bulk_si, P)
```

Then, we will apply random displacements to the atomic positions of a bulk Si supercell and vary the lattice constant using ASE atoms object. Please navigate to the `0.01A-1p` directory where you will find a Python script named `perturbations.py`. Random displacements within the defined maximum displacement will be added to the atomic positions, and random fractional changes within the defined maximum cell change will be applied to the lattice parameters. The Python script will generate a total of 100 frames and corresponding QE input files. In each frame, the cell and the atomic positions will be perturbed by a maximum of 1 % and 0.01 Å, respectively, from the ground state bulk Si structure. The degree of perturbation of the atomic positions and the cell for each frame follows a normal distribution.

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
