import numpy as np
import pytest

from src.sa import evaluate, is_feasible, school_loads, total_cost, validate

DIST = np.array([
    [1, 5, 9],
    [4, 2, 8],
    [7, 3, 6],
    [2, 9, 1],
])
CAP = np.array([2, 1, 2])


def test_total_cost_sums_assigned_distances():
    assert total_cost(DIST, [0, 1, 2, 2]) == 1 + 2 + 6 + 1


def test_total_cost_matches_loop():
    rng = np.random.default_rng(0)
    dist = rng.integers(0, 100, size=(30, 4))
    assign = rng.integers(0, 4, size=30)
    assert total_cost(dist, assign) == sum(dist[k, assign[k]] for k in range(30))


def test_school_loads():
    assert school_loads([0, 1, 2, 2], 3).tolist() == [1, 1, 2]
    assert school_loads([2, 2], 4).tolist() == [0, 0, 2, 0]


def test_valid_solution():
    assert validate(DIST, CAP, [0, 1, 2, 2]) is None
    assert is_feasible(DIST, CAP, [0, 1, 2, 0])
    assert evaluate(DIST, CAP, [0, 1, 2, 2]) == 10


def test_overfull_school_rejected():
    assert "pārpildīta" in validate(DIST, CAP, [0, 1, 1, 2])
    assert not is_feasible(DIST, CAP, [0, 0, 0, 2])
    with pytest.raises(ValueError):
        evaluate(DIST, CAP, [0, 1, 1, 2])


@pytest.mark.parametrize("assign", [[0, 1, 2], [0, 1, 2, 2, 0], [0, 1, 3, 2], [0, -1, 2, 2]])
def test_malformed_solution_rejected(assign):
    assert not is_feasible(DIST, CAP, assign)


def test_non_integer_indices_rejected():
    assert not is_feasible(DIST, CAP, [0.0, 1.0, 2.0, 2.0])
