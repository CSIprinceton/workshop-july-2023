## Deactivate the dp environment first.
conda deactivate
PW=/home/deepmd23admin/Softwares/QuantumEspresso/q-e-qe-7.0/bin/pw.x
W90=/home/deepmd23admin/Softwares/wannier90-3.1.0/wannier90.x
PWW90=/home/deepmd23admin/Softwares/QuantumEspresso/q-e-qe-7.0/bin/pw2wannier90.x
# kmesh=/home/deepmd23admin/Softwares/wannier90-3.1.0/utility/kmesh.pl

## run a SCF DFT calculation.  
mpirun  -np 6 $PW -input scf.in > scf.out

## run a non-SCF DFT calculation for getting complete information on orbitals
## usually we need a denser k-grid for wannierzation. But here we use the sparse 2X2X2 grid to save time.
mpirun  -np 6 $PW -input nscf.in > nscf.out

# generate .nnkp as the input of the postprocessing code pw2wannier90
$W90 -pp water

# produce the matrices needed for maximally localized wannierization .mmn, .amn, .eig…
mpirun  -np 6 $PWW90 < water.pw2wan > pw2wan.out

## On this virtual machine, we need to activate this conda environment for wannier90 to work with mpirun.
conda activate dp
## minimize the spread, calculate wannier function
/usr/bin/mpirun -np 6 $W90 water
 
