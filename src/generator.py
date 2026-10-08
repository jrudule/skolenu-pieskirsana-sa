"""Testu piemēru ģenerators un saglabāšana.

Skolēni un skolas tiek izvietoti nejauši kvadrātā [0, SIZE] x [0, SIZE];
attālumi ir noapaļoti Eiklīda attālumi (veseli skaitļi, lai precīzajam
algoritmam nebūtu noapaļošanas kļūdu). Skolu ietilpības ir nevienādas,
to summa ir ceil(slack * m) >= m.

Palaišana:  python -m src.generator   (izveido standarta piemērus instances/)
"""

import math
import os
from dataclasses import dataclass

import numpy as np

SIZE = 1000
# Standarta testu izmēri: (skolēni, skolas)
SIZES = [(20, 5), (100, 10), (500, 30), (1000, 50), (2000, 100)]
INSTANCE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "instances")


@dataclass
class Instance:
    name: str
    dist: np.ndarray        # (m, n) attālumi, int
    cap: np.ndarray         # (n,) ietilpības, int
    students_xy: np.ndarray  # (m, 2) koordinātas (tikai attēliem)
    schools_xy: np.ndarray   # (n, 2)

    @property
    def m(self):
        return self.dist.shape[0]

    @property
    def n(self):
        return self.dist.shape[1]


def random_capacities(m, n, rng, slack=1.1):
    """n ietilpības ar summu ceil(slack*m); katrai skolai vismaz 1 vieta."""
    total = max(m, n, math.ceil(slack * m))
    weights = rng.uniform(0.5, 1.5, size=n)
    return 1 + rng.multinomial(total - n, weights / weights.sum())


def generate(m, n, seed=0, slack=1.1, name=None):
    """Ģenerē piemēru ar m skolēniem un n skolām."""
    rng = np.random.default_rng(seed)
    students = rng.uniform(0, SIZE, size=(m, 2))
    schools = rng.uniform(0, SIZE, size=(n, 2))
    diff = students[:, None, :] - schools[None, :, :]
    dist = np.rint(np.sqrt((diff ** 2).sum(axis=2))).astype(np.int64)
    cap = random_capacities(m, n, rng, slack).astype(np.int64)
    return Instance(name or f"m{m}_n{n}", dist, cap, students, schools)


def save_instance(inst, directory=INSTANCE_DIR):
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, inst.name + ".npz")
    np.savez_compressed(path, dist=inst.dist.astype(np.int32), cap=inst.cap,
                        students_xy=inst.students_xy, schools_xy=inst.schools_xy)
    return path


def load_instance(path):
    data = np.load(path)
    name = os.path.splitext(os.path.basename(path))[0]
    return Instance(name, data["dist"].astype(np.int64), data["cap"].astype(np.int64),
                    data["students_xy"], data["schools_xy"])


def standard_instances(directory=INSTANCE_DIR, create=True):
    """Ielādē standarta piemērus; trūkstošos ģenerē un saglabā."""
    result = []
    for i, (m, n) in enumerate(SIZES):
        path = os.path.join(directory, f"m{m}_n{n}.npz")
        if os.path.exists(path):
            result.append(load_instance(path))
        elif create:
            inst = generate(m, n, seed=1000 + i)
            save_instance(inst, directory)
            result.append(inst)
        else:
            raise FileNotFoundError(path)
    return result


if __name__ == "__main__":
    for inst in standard_instances():
        print(f"{inst.name}: m={inst.m}, n={inst.n}, ietilpība={inst.cap.sum()}")
