import numpy as np
import ase.io
from ase.calculators.espresso import Espresso

################################
# QE options
################################
pseudopotentials = {'Si': 'Si_ONCV_PBE_sr.upf'}

input_qe = {
            'calculation':'scf',
            'outdir':'./',
            'pseudo_dir':'/home/ppiaggi/pseudos',
            'tprnfor':'.true.',
            'tstress':'.true.',
            'system':{
              'ecutwfc': 30,
              'input_dft': 'PBE',
             },
            'electrons':{
               'mixing_beta': 0.5,
               'electron_maxstep':1000,
             },
}

#################################
# LOAD
#################################
conf=ase.io.read('si.lammps-data',format='lammps-data',style='full')
initial_positions=conf.get_positions()
initial_cell=conf.get_cell()

################################
# Assign species
################################
print(np.array(conf.get_chemical_symbols()))
species=np.array(conf.get_chemical_symbols())
species=np.full(shape=species.shape,fill_value="Si")
conf.set_chemical_symbols(species)
print(np.array(conf.get_chemical_symbols()))

ase.io.write('pw-si.in',conf, format='espresso-in',input_data=input_qe, pseudopotentials=pseudopotentials,tstress=True, tprnfor=True)


