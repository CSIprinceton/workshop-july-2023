from ase.io import read, write
from ase.calculators.espresso import Espresso

# Specify the pseudopotentials for the elements
pseudopotentials = {'Si': 'Si_ONCV_PBE-1.0.upf'}

# Set up the input parameters for QE calculation
kpoints_list = []
n = 1  # Start with n = 1

while True:
    kpoints_list.append((n, n, n))
    n += 1
    if n == 7:  # Adjust the limit as needed
        break

offset = (1, 1, 1)

# Load the CIF file using ASE's read() function
bulk_si = read('Si.cif')

# Write the input file for QE calculation using ASE's write() function
for kpoints in kpoints_list:

    input_qe = {
        'calculation': 'scf',       
        'outdir': './',             
        'pseudo_dir': './',         
        'tprnfor': True,
        'tstress': True,
        'disk_io':'none',        
        'system': {
            'ecutwfc': 40,         
            'input_dft': 'PBE',     
        },
        'electrons': {
            'mixing_beta': 0.5,     
            'electron_maxstep': 1000
        },
    }

    kpoints_string = ''.join(str(x) for x in kpoints)

    write('pw-si-' + kpoints_string + '.in', bulk_si, format='espresso-in', input_data=input_qe,
          pseudopotentials=pseudopotentials, kpts=kpoints, koffset=offset, tstress=True, tprnfor=True)
