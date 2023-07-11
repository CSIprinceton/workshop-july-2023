# Basics of DFT Calculations with Quantum-ESPRESSO

Designed and written by Taehun Lee and Zachary K. Goldsmith, Princeton University

Hands-on sessions - Day 1 - July 11, 2023

Fundamentals of using Quantum-ESPRESSO for plane-wave DFT calculations of extended systems.

## Aims
This tutorial will demonstrate basic usage of the PW module of Quantum-ESPRESSO (QE), a leading open-source software for electronic structure, focusing on the practical utilities of key computational parameters and using crystalline Si as an example. This is intended as a straightforward tutorial for those who have not performed DFT calculations with QE in the past and will not cover the underlying physics and chemistry concepts. This exercise will cover how to benchmark and conduct ground state DFT simulations of periodic systems and extract results relevant to the training of deep neural network potentials.

## Objectives

This tutorial will cover the following:
- Necessary files and scripts for running QE calculations
- Anatomy of the QE input file 
- Submitting QE jobs
- Parsing and understanding QE output
- Exercises:
  - Benchmarking DFT parameters
  - Geometry relaxation

## Prerequisites

It is assumed that the participant has a general understanding of quantum mechanical calculations, proficiency with the linux command line, and basic level python scripting. Additional experience with plane-wave basis sets, crystal structure, and other solid-state physics concepts will also be helpful. This tutorial is furthermore written for Workshop participants who will have access to virtual machines which have QE v7.0 with GPU acceleration compiled. Instructions for downloading and compiling QE can be found at https://github.com/QEF/q-e.

The QE input and output files will be generated, maintained and parsed using [Atomic Simulation Environment (ASE)](https://wiki.fysik.dtu.dk/ase/index.html) which is written in the Python programming language with the aim of setting up, directing, and analyzing atomistic simulations.

## Running a DFT Calculation with QE

Running jobs with the PWSCF module of QE requires at minimum: 

1) The `pw.x` executable and its corresponding environment
2) Pseudopotentials in UPF format 
3) An input file

As mentioned previously, the `pw.x` executable and environment are readily available to participants with access to the VM. You will learn how to execute QE in the VM later. Otherwise, follow the instructions for downloading and compiling QE on your machine.

Different types of pseudopotentials and their underlying physics are beyond the scope of this tutorial, but there are many publically available pseudopotential libraries. This tutorial will utilize an [ONCV pseudopotential](http://quantum-simulation.org/potentials/sg15_oncv/upf/ "ONCV psp library") for Si optimized for PBE calculations. To retrieve this pseudopotential do the following:

```
wget http://quantum-simulation.org/potentials/sg15_oncv/upf/Si_ONCV_PBE-1.0.upf
```

Now we will begin by dissecting the QE input file using bulk Si as an example.

### Input file anatomy

The all-in-one guide for PWscf keywords is [here](https://www.quantum-espresso.org/Doc/INPUT_PW.html). This tutorial will address many of the most basic specifications.

Let's take a look at the file `si.in` located in this head directory, starting with the `&control` namelist:

```
 &control
   restart_mode = 'from_scratch',
   calculation  = 'scf',
   prefix       = 'si',
   outdir       = './',
   pseudo_dir = './',
   tprnfor = .true.,
 /
```
Start by noting the formatting of namelists; the `&` starts the namelist and the `/` terminates it. Keywords are separated by commas (and for our convenience but not necessarily, line breaks). `restart_mode = 'from_scratch',` implies that we are starting a calcualtion from scratch rather than restarting. `calculation  = 'scf',` entails that we are running a single-point self-consistent field (SCF) energy calculation. The prefix keyword sets the nomenclature for all output files. The `outdir` and `pseudo_dir` keywords specify the desired location of the outputs and pseudopotentials, respectively. In both cases, that will be the present directory `./`. Lastly, and importantly for DPMD applications `tprnfor = .true.,` will ensure that the atom-centered forces will be printed in the QE output.

Next, let's look at the `&system` namelist:

```
 &system
    ibrav=2,
    celldm(1) = 10.20,
    nat=2,
    ntyp=1,
    ecutwfc=24.0
    input_dft='pbe'
 /
```
`ibrav=2` indicates that our system has cubic FCC structure and symmetry, with `celldm(1)` defining the relevant lattice vector in au (bohr). QE's algorithms exploit crystal symmetries to accelerate calculations. 

`Xcrysden` can be used to visualize QE input and output files directly. With the corresponding symmetry, you can visualize both the conventional and primitive unit cells. To visualize QE input, you can bring the `si.in` into your local computer using `scp` command line.

```
xcrysden --pwi si.in
```

![image](https://user-images.githubusercontent.com/59068990/176943208-9a82fdb4-4c79-4393-872e-769a85220924.png)

There are several programs available for visualizing atomic structures. Here are some available options: [VESTA](https://jp-minerals.org/vesta/en/), [Ovito](https://www.ovito.org/about/), [ASE GUI viewer](https://wiki.fysik.dtu.dk/ase/ase/gui/gui.html) or python-based [NGL viewer](https://nglviewer.org). However, those programs cannot directly visualize the QE input and output files. To visualize the structures, you need to convert the QE input/output files to relevant structure file formats such as CIF, POSCAR (VASP), or XYZ.

NB: Crystal structure is beyond the scope of this tutorial, however, it is worth mentioning that non-crystalline (i.e. liquid, gaseous, interfacial) systems will use the `ibrav=0` option, in which the 3 x 3 lattice parameters must be specified explicitly. For an orthorhombic cell, all the off-diagonal elements would be zero. 

Straightforwardly, `nat` refers to the number of atoms and `ntyp` is the number of types of atoms. `ecutwfc` refers to the cutoff energy of the basis set planewaves. The higher this value, the more planewaves that are used, resulting in a slower, but more accurate calculation. We will explore the benchmarking of this value shortly. Lastly, `input_dft` indicates the DFT functional to be used in the calculation. The default value of this is the functional associated with the pseudopotential, so we wouldn't need to explicitly state this value in our case since we are using PBE, but it is included here to demonstrate where one would indicate the usage of e.g. SCAN functional.

Next is `&electrons`:

```
 &electrons
    conv_thr    = 1.D-6,
    mixing_beta = 0.5D0,
    startingwfc = 'atomic+random',
    startingpot = 'atomic',
 /
```

`conv_thr` is the energy convergence threshold for the SCF calculation. For the purposes of this tutorial we will leave it at the default. Lower values may be justifiable for larger systems further from equilibrium and/or to have an initial converged solution on which to improve. The `mixing_beta` parameter is an internal one related to the step-to-step perturbation of the trial wavefunction. We will not modify it in this tutorial but it is worth mentioning that smaller values typically yield slower but more stable paths to convergence. The `startingwfc` and `startingpot` are the initial wavefuncitons and potentials, respectively. We will not be modifying these keywords in this tutorial.

Lastly we come to the cards (note that these are not namelists and have different syntax) associated with the structure and k-points:

```
ATOMIC_SPECIES
 Si  28.086  Si_ONCV_PBE-1.0.upf
ATOMIC_POSITIONS (crystal)
 Si 0.00 0.00 0.00
 Si 0.25 0.25 0.25
K_POINTS automatic
 4 4 4 1 1 1
```
`ATOMIC_SPECIES` indicates the only species, Si, along with its atomic mass and the name of the corresponding pseudopotential file.

`ATOMIC_POSITIONS` is formatted in a familiar way: the type of atom and its 3D coordinates. In this input file we are exploiting the cubic symmetry so the positions are in units of the lattice vector, denoted by `alat`. This can be modified to `Angstrom` for non-symmetric systems. The two Si atoms form the basis of the cubic diamond crystal structure.

Last, `K_POINTS` refers to the sampling of the Brillouin Zone performed in the calculation. The technical details here are beyond the scope of this tutorial, but we will investigate the need to benchmark this value. 

### Input file generation using ASE-calculator
 You can access the Anaconda environment with pre-installed python 3.10 and ASE: 
```
conda activate dp
```

To generate the QE input file using the ASE calculator module, you need to load the relevant module. You can see more examples [here](https://wiki.fysik.dtu.dk/ase/ase/calculators/espresso.html#module-ase.calculators.espresso).

```python
from ase.io import read, write
from ase.calculators.espresso import Espresso

pseudopotentials = {'Si': 'Si_ONCV_PBE_sr.upf'}

# Define the input parameters for the QE calculation
input_qe = {
    'calculation': 'scf',             # Type of calculation (self-consistent field)
    'outdir': './',                   # Output directory
    'pseudo_dir': './',               # Directory for pseudopotential files
    'tprnfor': True,                  # Print forces in output
    'tstress': True,                  # Print stress tensor in output
    'disk_io': 'none',                # Disable disk I/O
    'system': {
        'ecutwfc': 40,                # Cutoff energy for wavefunctions (40 Ry)
        'input_dft': 'PBE',           # Exchange-correlation functional (PBE)
    },
    'electrons': {
        'mixing_beta': 0.5,           # Mixing parameter for electron density (0.5)
        'electron_maxstep': 1000      # Maximum number of electron iterations (1000)
    },
}

kpoints = (4, 4, 4)                    # K-point mesh size
offset = (1, 1, 1)                     # Offset for the k-point mesh
```
The given code defines two dictionaries, `pseudopotentials` and `input_qe`, which are used to set up parameters for a QE calculation, as explained previously. These dictionaries provide the necessary input parameters for configuring a QE calculation using the specified pseudopotentials and system parameters. It's important to note that the code does not explicitly define variables for the default settings. The variables `kpoints` and `offset` are used to define the k-points grid in the calculations.

Instead of manually setting the crystal structure, you can utilize the ASE Atoms object, which stores information about the chemical and crystal structure of a system. By defining the ASE Atoms object, you can automatically set QE flags related to the chemical and crystal structure, such as `nat`, `ntyp`, `ibrav`, and generate the necessary `ATOMIC_SPECIES` and `ATOMIC_POSITIONS` cards in the QE input file. 

You can define an ASE Atoms object for bulk Si by either manually setting the structure or loading a CIF file or relevant structure files. In this case, we will load a CIF file obtained from the [Materials Project](https://next-gen.materialsproject.org) or other relevant materials database.

```python
from ase.io import read

# Load the CIF file using ASE's read() function
bulk_si = read('Si.cif')
```

Now, you can generate the QE input file using the provided dictionary and variables:
```python
ase.io.write('pw-si.in', bulk_si, format='espresso-in',input_data=input_qe, pseudopotentials=pseudopotentials, kpts=kpoints, koffset=offset)
```
This code will generate the QE input file named `pw-si.in` based on the ASE Atoms object `bulk_si`, using the specified input parameters, pseudopotentials, k-points, and offset values. You can find the compiled Python script named `bulk_si.py` in the tutorial folder. You can run the script by typing: 

```
python bulk_si.py
```

### Running QE jobs

With all of our necessary components ready, we can now proceed to run a simple QE job. In the VM, QE v7.0 is compiled and the executable is located at `/home/deepmd23admin/Softwares/QuantumEspresso/q-e-qe-7.0/bin/pw.x`. Thus, you can run the simple calculation by typing:

```shell
conda deactivate
export PW=/home/deepmd23admin/Softwares/QuantumEspresso/q-e-qe-7.0/bin/pw.x
mpirun -np 6 $PW -input pw-si.in > pw-si.out
```
Once the calculation is completed, you will find the output written to the file `pw-si.out`.

### Parsing QE output

So, what happened when we ran the job? In summary, QE iteratively converged the eigenvectors and eigenvalues of the Si system starting from an initial guess. Before looking at details of the output file, let's check if the calculation has completed successfully. You can determine this by checking the end of the output file (`pw-si.out`) for the following completion message.

```
=------------------------------------------------------------------------------=
   JOB DONE.
=------------------------------------------------------------------------------=
```

To see the total energy of the self-consistent field (SCF) calculation, you can open the output file and locate the character `!`. The lines following this total energy will provide information about its constituent terms, the number of iterations required for convergence, and the forces acting on each atom. In the case of Si at equilibrium, the forces should be zero. Note the structure is not at equilibrium since it was taken from the database, and not obtained with DFT structural optimization calculation.

```
!    total energy              =     -63.05587754 Ry
     Harris-Foulkes estimate   =     -63.05587751 Ry
     estimated scf accuracy    <       0.00000049 Ry

     The total energy is the sum of the following terms:

     one-electron contribution =      18.77798849 Ry
     hartree contribution      =       4.42766960 Ry
     xc contribution           =     -19.23489981 Ry
     ewald contribution        =     -67.02663583 Ry

     convergence has been achieved in   5 iterations

     Forces acting on atoms (cartesian axes, Ry/au):

     atom    1 type  1   force =    -0.00000429   -0.00000429    0.00000429
     atom    2 type  1   force =     0.00000000    0.00000000    0.00000000
     atom    3 type  1   force =    -0.00000429    0.00000429   -0.00000429
     atom    4 type  1   force =     0.00000000    0.00000000    0.00000000
     atom    5 type  1   force =     0.00000429   -0.00000429   -0.00000429
     atom    6 type  1   force =     0.00000000    0.00000000    0.00000000
     atom    7 type  1   force =     0.00000429    0.00000429    0.00000429
     atom    8 type  1   force =     0.00000000    0.00000000    0.00000000

     Total force =     0.000015     Total SCF correction =     0.000689
     SCF correction compared to forces is large: reduce conv_thr to get better values
```

Let's also look at the progression of the calculation to convergence with:

```
grep "total energy              =" pw-si.out
```

You should see the energy decrease monotonically to the final energy. 

### Parsing QE output using ASE-calculator

You can also parse important physical and chemical quantities of QE output using the ASE module as follows:

```python
## read QE output file
bulk_si_out = read('pw-si.out', format='espresso-out')  # Returns an Atoms object

## Print physical and chemical quantities
print('Atomic positions:   in angstrom')
print(bulk_si_out.get_positions())
print('Lattice vector  :   ', bulk_si_out.get_cell())
print('Total energy    :   ', round(bulk_si_out.get_potential_energy(),5), 'eV')
```
You can run the script by typing `python output_parse.py`. ASE atoms object returns the total energy of the system in electron volts (eV), not Ry. Using the ASE module, you can access various physical quantities and chemical properties stored in the ASE calculator, such as volume, magnetic moment, eigenvalues, and occupations. Please explore the ASE documentation for a comprehensive list of available methods to access different physical and chemical properties stored in the ASE calculator.

## Exercises: Benchmarking and Geometry

### Benchmarking DFT protocol

It is critical that one benchmarks their DFT protocol, especially given that the accuracy of the DFT calculation is ultimately what a machine-learned potential will achieve with sufficient training. Here we will demonstrate how to benchmark two of the most important aspects of QE DFT: `ecutwfc` and the number of k-points.

1. `Ecutwfc`: In plane-wave DFT calculations, one should use a plane-wave energy cutoff that is sufficiently high such that the computed energy for a sample system is stable with respect to this cutoff. In other words, we are exploring how the number of plane-waves (basis set size) affects the energy and time to solution. Move to the directory `ecut`. Therein you will find a Python script, `ecut.py`. Then, run the script to generate QE input files with different plane-wave energy cutoff values ranging from 10 to 60 Ry. The script sets up a range of cutoff energies for wavefunctions using the range() function and then loops over the cutoff energies and generates QE input files with different plane-wave energy cutoff values. Following is the highlight of the important part.

```python
# Set up the range of cutoff energies for wavefunctions
wfcs = range(10, 70, 10)

# Loop over the cutoff energies and generate QE input files
for wfc in wfcs:
    input_qe = {
 	...
        'system': {
            'ecutwfc': wfc,         
        },
 	...
    }
    write('pw-si-' + str(wfc) + '.in', bulk_si, format='espresso-in', input_data=input_qe,
          pseudopotentials=pseudopotentials, kpts=kpoints, koffset=offset, tstress=True, tprnfor=True)
```
Accordingly, you should make a change to the command line for the QE executable using loop:

```shell
conda deactivate
export PW=/home/deepmd23admin/Softwares/QuantumEspresso/q-e-qe-7.0/bin/pw.x
for i in `seq 10 10 60`
do
        mpirun -np 6 $PW -input pw-si-$i.in > pw-si-$i.out
done
```

After the completion of calculations, let's examine the computed energies and their convergence. It is important to note that the energy decreases with increasing `ecutwfc` in the QE input file (or `wfc` variable in the Python file), but with diminishing returns at higher values. A properly benchmarked calculation would involve using an `ecutwfc` value beyond the point where the energy doesn't change significantly. 

To visualize this trend, you can plot `ecutwfc` versus `total energy` using a simple IPython script (`plot.ipython`). For each cutoff energy, it reads the output file (`pw-si-<wfc>.out`) using ASE's read() function, returning the total energy. It will plot the energies versus the cutoff energies and save the plot as an image file (`ecut.png`). You can run it using Jupyter notebooks run on the virtual machine and opened in your local browser. In order to do this, first execute on the **remote machine** (if you are in `.../ecut` we recommend first doing a `cd ../`):

```
conda activate dp
nohup jupyter notebook --port=2333 &
```
and then run in your **local machine**:
```
ssh -N -f -L localhost:2333:localhost:2333 -p <port> <username>@<remote-machine-address>
```
At the end of the ```nohup.out``` file you will find a link that you can copy and then paste into your browser. 

<p float="left">
  <img src="https://github.com/CSIprinceton/workshop-july-2023/blob/6ed432411c4285a8dea9a77ce027c485d3e09b71/hands-on-sessions/day-1/2-quantum-espresso/ecut.png" width="400"> 
</p>

Another quick way to see the energies from each calculation is to do `grep ! *.out`.

2. K-points: Similarly, it is important to achieve convergence of energy by sampling an appropriate number of k-points in a periodic system. Please navigate to the `kpoints` directory where you will find a Python script named `kp.py`. This script generates a series of input files with increasing k-grid densities, ranging from 1 x 1 x 1 to 6 x 6 x 6 by typing `python kp.py'. Make sure to modify the QE executable command line:

```shell
conda deactivate
export PW=/home/deepmd23admin/Softwares/QuantumEspresso/q-e-qe-7.0/bin/pw.x
for i in `seq 1 1 6`
do
        mpirun -np 6 $PW -input pw-si-$i$i$i.in > pw-si-$i$i$i.out
done
``` 

Upon completion of the calculations, let's analyze the computed energies and their convergence with respect to the k-grid. 

<p float="left">
  <img src="https://github.com/CSIprinceton/workshop-july-2023/blob/6ed432411c4285a8dea9a77ce027c485d3e09b71/hands-on-sessions/day-1/2-quantum-espresso/kpoint.png" width="400"> 
</p>

Notice that initially, the energy exhibits a significant decrease as the k-point sampling becomes denser, indicating a higher level of accuracy. However, beyond a certain point, typically around 3 x 3 x 3, the energy converges and shows minimal changes with further increases in the k-point sampling. It is important to identify this converged region and choose a k-point sampling that lies within it for efficient and reliable calculations.

### Structural optimization 

There are two types of structural optimization calculations in QE:
- `relax`: where only the atomic positions are allowed to vary
- `vc-relax`: which allows for varying both the atomic positions and lattice constants.

Later, we will perturb the structure of the ground state bulk Si unit cell to utilize its structure and energy for training deep neural network potentials. However, before that, we need to obtain the ground state of the bulk Si using a `vc-relax` optimization. When comparing the `vc-relax` input file to the SCF input file, you will notice a few differences. First, in the `&control` namelist, it is specified that this is a relax calculation with `calculation = 'vc-relax'`. The `forc_conv_thr` parameter sets the force convergence threshold for the calculation. Additionally, a relax calculation requires the inclusion of additional `&ions` and `&cell` namelists. The BFGS algorithm is the default relaxation algorithm used.

```
&CONTROL
   calculation      = 'vc-relax'
   forc_conv_thr    = 1.0D-4
/
&IONS
   ion_dynamics     = 'bfgs'
/
&CELL
   cell_dynamics    = 'bfgs'
/
```

Here, the modified part of Python script for the `vc-relax` calculation is highlighted:

```python
input_qe = {
    'calculation': 'vc-relax',
    'forc_conv_thr': 1.0e-4,
    'ions': {
        'ion_dynamics': 'bfgs',
    },
    'cell': {
        'cell_dynamics': 'bfgs',
    },
}
```

Please navigate to the `vcopt` directory and refer to the python script `bulk_si_vc-relax.py`. This script will generate a new input file named `pw-si-vc_relax.in`. You can use this input file to perform the structural optimization and obtain the equilibrium structural parameters of bulk Si as follows:
```shell
conda deactivate
export PW=/home/deepmd23admin/Softwares/QuantumEspresso/q-e-qe-7.0/bin/pw.x
mpirun -np 6 $PW -input pw-si-vc_relax.in > pw-si-vc_relax.out
```

In a relax calculation, an electronic SCF is converged for every ionic step to reduce the forces below the specified threshold. Let's examine the convergence of electronic energies and the reduction of forces during the relax calculation using the following command lines:
```
Energies: grep ! pw-si-vc_relax.out
Forces:   grep "Total force" pw-si-vc_relax.out
```

Once the calculation is completed, you should compare the obtained lattice constant with the literature value (5.43 Å) and check the forces on the atoms to ensure they approach zero. You can use the `output_parse.py` script for this purpose. Additionally, the script will generate structure files (cif and xyz) that can be visualized in different programs such as VESTA and OVITO. 

```python
python ouput_parse.py
```

To visualize the structural relaxation as an animation, you can also use `Xcrysden` by bringing the output into your local computer with the `scp` command. 

```
xcrysden --pwo pw-si-vc_relax.out
```

In the `xcrysden` GUI, select to display all coordinates as an animation. You can also measure the Si-Si distance at the beginning and end of the calculation using the `Distance` tool located at the bottom of the `xcrysden` GUI. Select the two atoms, then click `Done`.

### Additional considerations and links

- [LibXC](https://gitlab.com/libxc/libxc/-/releases) is the library QE uses for meta-GGA, hybrid, etc. functionals. Much of the pioneering DPMD work on water was trained with the SCAN functional, which requires LibXC to run in QE.
- [Materials Cloud](https://www.materialscloud.org/home) provides a tool to generate input files for the QE PWscf code and visualize corresponding structures. This platform also offers a standard solid-state pseudopotential (SSSP) library optimized for precision or efficiency. It provides convergence results for various physical and chemical quantities, such as phonons, cohesive energy, and pressure, as a function of the wavefunction cutoff for different pseudopotentials which will reduce the computational resources for the convergence tests.
