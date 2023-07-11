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
for i in range(100):
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
