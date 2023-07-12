from pathlib import Path
from ase.io import read, write

def read_structures(w, number=100):
    """"""
    frames = []
    for i in range(number):
        print(f"read structures {i}")
        atoms = read(Path(w)/f"pw-si-{i}.out", format="espresso-out")
        frames.append(atoms)

    return frames

# - read structures
print(Path.cwd())
frames = read_structures(Path.cwd())
write("./configurations.xyz", frames)

