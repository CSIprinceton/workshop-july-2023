# Basics of preparing training data

Designed and written by Pablo Piaggi and Taehun Lee, Princeton University

Hands-on sessions - Day 1 - July 11, 2023

## Aims and Objectives

This tutorial offers an overview of the process of preparing training data for deep potential (DP) models. It focuses on two techniques: random perturbation and molecular dynamics simulations.

## Backgrounds

As one branch of machine learning-based interatomic potential model training, DP model fits interatomic potential energy surface, using deep neural networks, typically from datasets calculated by DFT. The process of training DP model involves the following steps: (1) exploration and labeling: constructing a database consisting of DFT calculations by exploring configuration spaces and labeling the dataset for DP model training, (2) training and validating DP models using the [deepMD-kit](https://docs.deepmodeling.com/projects/deepmd/en/master/), and (3) performing molecular dynamics simulations using the DP model interfaced with [LAMMPS](https://www.lammps.org/#gsc.tab=0).

<div align="center">
<img src="https://github.com/CSIprinceton/workshop-july-2023/blob/cf3e79c9402f39423c75a61cbed22e4a14dc6313/hands-on-sessions/day-1/3-preparing-training-data/protocol_sampling.png" width="1000"> 
</div>

*The left-hand side is taken from the deepMD-kit manual [1]() and the right-hand side of the figure is drawn in the style of ref. [2]() and [3]().*
<br/><br/>

In particular, for step (1), the construction of the training dataset can be done in various ways, depending on the purpose of the final DP model. The database consists of multiple frames, with each frame containing chemical and configurational information, DFT-computed forces, and potential energy. It is important to note that developing and utilizing DP can be more challenging when the system has diverse element compositions or includes interfaces (or phase boundaries) such as vacuum/solid and water/solid. Exploring and labeling such complex systems may require additional algorithms or techniques, such as global optimization or enhanced sampling methods (shown in the above figure).

## Exercises
In this tutorial, we will walk you through the process of preparing a training dataset for `DeepMD-kit`. You will sample a set of bulk Si structures using two different techniques to create a diverse set of starting configurations: 
- Manual construction using random perturbation and vibrations of atoms and cells from their equilibrium positions.
- Molecular dynamics simulations.

Using the sampled structures, you will perform DFT calculations with QE to obtain the potential energy and atomic forces for each structure (or frame). The calculated data, including the potential energy and atomic forces, will be converted into the appropriate input formats required by `DeepMD-kit` for subsequent analysis and training of interatomic potential models.

### Random perturbation
**1. Exploration**: Please navigate to the `perturbations-si-64/0.01A-1p` directory where you will find a Python script named `perturbations.py`. Using this script, first, the structurally optimized structure of bulk Si with 8 atoms will be read as ASE atoms object. Then the supercell will be constructed by expanding the unit cell using (2 x 2 x 2) transformation vector which yields the supercell with 64 atoms.

```python
from ase.build import make_supercell

bulk_si = ase.io.read('../pw-si-vc_relax.out',format='espresso-out')
P = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
conf = make_supercell(bulk_si, P)
```

Then, you will apply random displacements to the atomic positions of a bulk Si supercell and vary the lattice parameters using ASE atoms object. Random displacements within the defined maximum displacement will be added to the equilibrium atomic positions, and random fractional changes within the defined maximum cell change will be applied to the lattice parameters. The Python script will generate a total of `100 frames` and corresponding QE input files. In each frame, the cell and the atomic positions will be perturbed by a maximum of `1 %` and `0.01 Å`, respectively, from the ground state bulk Si structure. The degree of perturbation of the atomic positions and the cell for each frame follows a normal distribution.

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

Accordingly, you should make a change in the job script file as well:
```
for i in `seq 0 99`
do
	srun --mpi=pmi2 \
	singularity run --nv \
     	/scratch/gpfs/ppiaggi/Simulations/QuantumEspressoGPU/quantum_espresso_qe-7.1.sif \
     	pw.x -input pw-si-$i.in -npool 2 > pw-si-$i.out
done
```
Now you can perform SCF DFT calculation for each frame to evaluate the forces and energy. Let's play with the `max_displacement` and `max_cell_change` variables by constructing the different datasets to sample enough chemical spaces (refer to `0.05A-2p`, `0.1A-3p`, `0.2A-5p` directories).


<br/>


**2. Labeling:** Now that you have generated a set of atomic configurations from the exploration step and computed their energy and forces from DFT calculations, the next step is to label these configurations. This labeling step involves extracting raw data from the PW outputs and converting them into the input format required by `deepMD-kit` for training. A full list of these files can be found [here](https://github.com/deepmodeling/deepmd-kit/blob/master/doc/data/system.md). The following is a description of the basic `deepMD-kit` input formats:

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

You can parse the atomic structures, potential energy, and atomic forces from QE outputs using the ASE calculator and numpy-based python script named `get_raw.py`.

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

for i in range(101):
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
Let's verify if this script successfully generates the files `coord.raw`, `energy.raw`, `force.raw`, `virial.raw`, `box.raw`, and `type.raw`. It's important to note that while the raw format is not directly supported for training, NumPy and HDF5 binary formats are supported. To convert the prepared raw files to the NumPy, you can utilize the provided tool `raw_to_set.sh`.

### 2. MD simulations
t.b.d.

## Additional considerations and references
- [DP-GEN](): A user-friendly and **automatic** Python software for generating accurate and efficient DP models dependent on `DeePMD-kit`.
- [Dpdata](https://github.com/deepmodeling/dpdata): A python package for manipulating data formats of software, including DeePMD-kit, QE, VASP, LAMMPS, GROMACS, Gaussian, and CP2K.
