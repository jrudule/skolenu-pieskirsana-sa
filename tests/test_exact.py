import numpy as np
import pytest

from src.exact import brute_force, lower_bound, min_cost_assignment
from src.generator import generate, random_capacities
from src.sa import State, is_feasible, simulated_annealing, total_cost


def tiny_instance(seed):
    """Ļoti mazs piemērs ar ciešām ietilpībām: m <= 7, n <= 3."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 4))
    m = int(rng.integers(n, 8))
    dist = rng.integers(0, 20, size=(m, n))
    cap = random_capacities(m, n, rng, slack=float(rng.choice([1.0, 1.2])))
    return dist, cap


@pytest.mark.parametrize("seed", range(40))
def test_exact_matches_brute_force(seed):
    dist, cap = tiny_instance(seed)
    _, bf_cost = brute_force(dist, cap)
    assign, cost = min_cost_assignment(dist, cap)
    assert is_feasible(dist, cap, assign)
    assert cost == total_cost(dist, assign) == bf_cost


def test_brute_force_small_known_case():
    dist = np.array([[1, 10], [1, 10], [10, 1]])
    assign, cost = brute_force(dist, [1, 2])
    assert cost == 1 + 10 + 1


@pytest.mark.parametrize("seed", range(5))
def test_sa_finds_optimum_on_tiny(seed):
    dist, cap = tiny_instance(100 + seed)
    _, opt = brute_force(dist, cap)
    assert simulated_annealing(dist, cap, seed=seed).cost == opt


@pytest.mark.parametrize("m,n,seed", [(100, 10, 0), (300, 20, 1)])
def test_exact_is_locally_optimal_and_bounds_hold(m, n, seed):
    inst = generate(m, n, seed, slack=1.05)
    assign, opt = min_cost_assignment(inst.dist, inst.cap)
    assert is_feasible(inst.dist, inst.cap, assign)
    assert lower_bound(inst.dist) <= opt
    # Nevienam SA gājienam nedrīkst būt Δ < 0 attiecībā pret optimumu.
    st = State(inst.dist, inst.cap, assign)
    for k in range(m):
        for j in range(n):
            if st.can_relocate(k, j):
                assert st.delta_relocate(k, j) >= 0
    d = inst.dist
    a = assign
    swap = d[:, a].T + d[:, a] - d[np.arange(m), a][:, None] - d[np.arange(m), a][None, :]
    assert swap.min() >= 0
    assert simulated_annealing(inst.dist, inst.cap, seed=0).cost >= opt


def test_lower_bound_is_tight_without_capacity_limits():
    inst = generate(80, 6, seed=3)
    _, opt = min_cost_assignment(inst.dist, np.full(6, 80))
    assert opt == lower_bound(inst.dist)
