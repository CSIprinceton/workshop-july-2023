ln -s ../train_energy_model/frozen_model.pb frozen_model.pb

conda activate dp
lmp -v TEMP 330 -v PRES 1.0 -in in.lammps > thermo.log
