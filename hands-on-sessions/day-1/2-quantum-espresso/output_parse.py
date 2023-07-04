from ase.io import read, write

## read QE output file
bulk_si_out = read('pw-si.out', format='espresso-out')  # Returns an Atoms object

## Print physical and chemical quantities
print('Atomic positions:   in angstrom')
print(bulk_si_out.get_positions())
print('Lattice vector  :   ', bulk_si_out.get_cell())
print('Total energy    :   ', round(bulk_si_out.get_potential_energy(),5), 'eV')  ##in eV
