import copy
import pathlib
import json
import numpy as np

template = "./input.json"
with open(template, "r") as fopen:
    temp_dpconfig = json.load(fopen)

num_models = 4
# np.random.randint(0, 10000, 4)
rng = np.random.default_rng(seed=1112)
random_seeds = rng.integers(0, 10000, 4).tolist()
print(random_seeds)

for i, seed in enumerate(random_seeds):
    # - update seed
    curr_dpconfig = copy.deepcopy(temp_dpconfig)
    curr_dpconfig["model"]["descriptor"]["seed"] = seed
    curr_dpconfig["model"]["fitting_net"]["seed"] = seed
    curr_dpconfig["training"]["seed"] = seed
    # - mkdir and write
    wdir = pathlib.Path(f"{i+1}")
    wdir.mkdir() # If dir exists, an error will be raised.
    with open(wdir/"./input.json", "w") as fopen:
        json.dump(curr_dpconfig, fopen, indent=4)
