from ase.io import read, write
from ase.calculators.espresso import Espresso

# Specify the pseudopotentials for the elements
pseudopotentials = {'Si': 'Si_ONCV_PBE-1.0.upf'}

# Set up the input parameters for QE calculation
kpoints = (4, 4, 4)
offset = (1, 1, 1)

# Load the CIF file using ASE's read() function
bulk_si = read('Si.cif')

# Range of cutoff energies for wavefunctions
wfcs = range(10, 70, 10)  

# Write the input file for QE calculation using ASE's write() function
for wfc in wfcs:

    input_qe = {
        'calculation': 'scf',       
        'outdir': './',             
        'pseudo_dir': './',         
        'tprnfor': True,
        'tstress': True,
        'disk_io':'none',        
        'system': {
            'ecutwfc': wfc,         
            'input_dft': 'PBE',     
        },
        'electrons': {
            'mixing_beta': 0.5,     
            'electron_maxstep': 1000
        },
    }

    write('pw-si-' + str(wfc) + '.in', bulk_si, format='espresso-in', input_data=input_qe,
          pseudopotentials=pseudopotentials, kpts=kpoints, koffset=offset, tstress=True, tprnfor=True)
