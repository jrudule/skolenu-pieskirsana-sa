import numpy as np
import pytest

from src.generator import generate, load_instance, save_instance
from src.sa import greedy_initial, is_feasible, total_cost


@pytest.mark.parametrize("m,n,seed", [(20, 5, 0), (100, 10, 1), (500, 30, 2)])
def test_generator_shapes_and_capacity(m, n, seed):
    inst = generate(m, n, seed)
    assert inst.dist.shape == (m, n)
    assert inst.cap.shape == (n,)
    assert inst.cap.min() >= 1
    assert inst.cap.sum() >= m
    assert np.issubdtype(inst.dist.dtype, np.integer) and inst.dist.min() >= 0


def test_generator_is_deterministic():
    a, b = generate(50, 7, seed=3), generate(50, 7, seed=3)
    assert np.array_equal(a.dist, b.dist) and np.array_equal(a.cap, b.cap)
    assert not np.array_equal(a.dist, generate(50, 7, seed=4).dist)


def test_save_and_load_roundtrip(tmp_path):
    inst = generate(30, 4, seed=5)
    loaded = load_instance(save_instance(inst, tmp_path))
    assert loaded.name == inst.name
    assert np.array_equal(loaded.dist, inst.dist)
    assert np.array_equal(loaded.cap, inst.cap)


@pytest.mark.parametrize("m,n,seed", [(20, 5, 0), (100, 10, 1), (500, 30, 2)])
def test_greedy_is_feasible(m, n, seed):
    inst = generate(m, n, seed)
    assert is_feasible(inst.dist, inst.cap, greedy_initial(inst.dist, inst.cap))


def test_greedy_picks_nearest_free_school():
    dist = np.array([[1, 2, 3], [1, 2, 3], [1, 3, 2]])
    cap = np.array([1, 1, 1])
    assert greedy_initial(dist, cap).tolist() == [0, 1, 2]


def test_greedy_without_capacity_limits_is_optimal():
    inst = generate(60, 6, seed=7)
    cap = np.full(6, 60)
    assign = greedy_initial(inst.dist, cap)
    assert total_cost(inst.dist, assign) == inst.dist.min(axis=1).sum()


def test_greedy_rejects_insufficient_capacity():
    with pytest.raises(ValueError):
        greedy_initial(np.ones((3, 2)), np.array([1, 1]))
