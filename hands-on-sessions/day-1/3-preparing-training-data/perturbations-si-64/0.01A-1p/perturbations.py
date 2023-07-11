import numpy as np
from ase.io import read, write
from ase.calculators.espresso import Espresso
from ase.build import make_supercell

################################
# Parameters
################################
max_displacement=0.01 # Maximum displacement in angstrom
max_cell_change=0.01 # Maximum fractional change in cell

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
              'occupations': 'smearing',
              'smearing': 'fermi-dirac',
              'degauss': 0.01,
             },
            'electrons':{
               'mixing_beta': 0.5,
               'electron_maxstep':1000,
             },
}

#################################
# LOAD and make supercell
#################################
bulk_si = read('../pw-si-vc_relax.out',format='espresso-out')
P = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
conf = make_supercell(bulk_si, P)

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
    write('pw-si-' + str(i) + '.in',conf, format='espresso-in',input_data=input_qe, pseudopotentials=pseudopotentials)
