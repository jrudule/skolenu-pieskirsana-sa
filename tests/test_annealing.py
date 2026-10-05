import numpy as np
import pytest

from src.generator import generate
from src.sa import greedy_initial, is_feasible, simulated_annealing, total_cost


@pytest.mark.parametrize("m,n,seed", [(20, 5, 0), (100, 10, 1)])
def test_sa_returns_feasible_and_not_worse_than_greedy(m, n, seed):
    inst = generate(m, n, seed)
    res = simulated_annealing(inst.dist, inst.cap, seed=seed)
    assert is_feasible(inst.dist, inst.cap, res.assign)
    assert res.cost == total_cost(inst.dist, res.assign)
    assert res.cost <= total_cost(inst.dist, greedy_initial(inst.dist, inst.cap))
    assert res.cost <= res.initial_cost


def test_sa_is_reproducible_with_seed():
    inst = generate(80, 8, seed=4)
    a = simulated_annealing(inst.dist, inst.cap, seed=7)
    b = simulated_annealing(inst.dist, inst.cap, seed=7)
    assert a.cost == b.cost and np.array_equal(a.assign, b.assign)


def test_sa_history_and_cooling():
    inst = generate(50, 5, seed=2)
    res = simulated_annealing(inst.dist, inst.cap, seed=0, alpha=0.9, t0=100.0, t_end=1.0,
                              iters_per_temp=50)
    temps = [h[1] for h in res.history]
    assert np.allclose(np.diff(temps) / np.array(temps[:-1]), -0.1)
    assert temps[-1] <= 1.0 < temps[-2]
    best = [h[3] for h in res.history]
    assert all(b2 <= b1 for b1, b2 in zip(best, best[1:]))
    assert res.iterations == 50 * (len(res.history) - 1)


def test_sa_improves_tight_instance():
    # Ar ierobežotām ietilpībām alkatīgais risinājums nav optimāls.
    inst = generate(200, 10, seed=11, slack=1.0)
    res = simulated_annealing(inst.dist, inst.cap, seed=0)
    assert res.cost < res.initial_cost
