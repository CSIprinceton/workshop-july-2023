import argparse

import numpy as np

from ase.io import read, write
from ase.calculators.calculator import all_changes
from ase.calculators.mixing import LinearCombinationCalculator

from deepmd.calculator import DP

"""Taking from https://github.com/hsulab/GDPy
"""

class CommitteeCalculator(LinearCombinationCalculator):

    def __init__(self, calcs, use_avg=False, save_atomic=True, ddof=0, atoms=None):
        """Init the committee calculator.

        Args:
            calcs: ASE calculators.
            use_avg: Whether use average results instead of those by the first calc.
            save_atomic: Whether save atomic deviation.
            ddof: Dela Degrees of Freedom that affects the deviations of results.

        """
        weights = np.ones(len(calcs))
        if use_avg:
            weights = weights / np.sum(weights)
        else:
            weights[1:] = 0.
        self.ddof = ddof
        self.save_atomic = save_atomic

        super().__init__(calcs, weights, atoms)

        return

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        """"""
        prev_calc = atoms.calc
        atoms.calc = None

        # TODO: set wdir for each subcalc
        super().calculate(atoms, properties, system_changes)
        atoms.calc = prev_calc

        self._compute_deviation(atoms, properties)

        return
    
    def _compute_deviation(self, atoms, properties):
        """Compute the RMSE deviation of calculator properties."""
        if "energy" in properties:
            tot_energies = np.array([c.results["energy"] for c in self.calcs])
            self.results["devi_te"] = np.sqrt(np.var(tot_energies, ddof=self.ddof))

        if "forces" in properties:
            cmt_forces = np.array([c.results["forces"].flatten() for c in self.calcs])
            frc_devi = np.sqrt(np.var(np.array(cmt_forces), axis=0))
            self.results["max_devi_f"] = np.max(frc_devi)
            self.results["min_devi_f"] = np.min(frc_devi)
            self.results["avg_devi_f"] = np.mean(frc_devi)
            if self.save_atomic:
                self.results["devi_f"] = np.reshape(frc_devi, (-1,3))
        
        # TODO: atomic energies?

        return

parser = argparse.ArgumentParser()
parser.add_argument(
    "--structure"
)
parser.add_argument(
    "--model", nargs="*"
)
args = parser.parse_args()

calcs = []
for m in args.model:
    curr_calc = DP(model=m, type_dict={"Si": 0})
    calcs.append(curr_calc)

calc = CommitteeCalculator(calcs)

init_structures = read(args.structure, ":")

deviations = []
for a in init_structures:
    calc.reset()
    a.calc = calc
    _ = a.get_forces()
    deviations.append(calc.results["max_devi_f"])

np.savetxt("./max_devi_f.txt", deviations, fmt="%12.8f")

