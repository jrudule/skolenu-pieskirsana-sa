import numpy as np
import pytest

from src.generator import generate
from src.sa import State, greedy_initial, school_loads, total_cost, validate


def make_state(m=60, n=6, seed=0):
    inst = generate(m, n, seed, slack=1.3)
    return inst, State(inst.dist, inst.cap, greedy_initial(inst.dist, inst.cap))


def check_consistent(inst, st):
    """Visi inkrementāli uzturētie lielumi sakrīt ar pilnu pārrēķinu."""
    assign = st.assignment()
    assert validate(inst.dist, inst.cap, assign) is None
    assert st.cost == total_cost(inst.dist, assign)
    assert st.load == school_loads(assign, inst.n).tolist()
    for j in range(inst.n):
        assert sorted(st.members[j]) == np.flatnonzero(assign == j).tolist()
        for p, k in enumerate(st.members[j]):
            assert st.pos[k] == p


def test_relocate_delta_matches_full_recompute():
    inst, st = make_state()
    rng = np.random.default_rng(1)
    done = 0
    for _ in range(2000):
        k, j = int(rng.integers(inst.m)), int(rng.integers(inst.n))
        if not st.can_relocate(k, j):
            continue
        before = total_cost(inst.dist, st.assignment())
        d = st.delta_relocate(k, j)
        st.relocate(k, j)
        assert total_cost(inst.dist, st.assignment()) - before == d
        done += 1
    assert done > 100
    check_consistent(inst, st)


def test_swap_delta_matches_full_recompute():
    inst, st = make_state(seed=2)
    rng = np.random.default_rng(3)
    for _ in range(2000):
        k1, k2 = (int(x) for x in rng.integers(inst.m, size=2))
        before = total_cost(inst.dist, st.assignment())
        d = st.delta_swap(k1, k2)
        st.swap(k1, k2)
        assert total_cost(inst.dist, st.assignment()) - before == d
    check_consistent(inst, st)


@pytest.mark.parametrize("seed", range(5))
def test_mixed_moves_keep_state_consistent(seed):
    inst, st = make_state(m=40, n=5, seed=seed)
    rng = np.random.default_rng(seed)
    for _ in range(3000):
        if rng.random() < 0.5:
            k, j = int(rng.integers(inst.m)), int(rng.integers(inst.n))
            if st.can_relocate(k, j):
                st.relocate(k, j)
        else:
            st.swap(*(int(x) for x in rng.integers(inst.m, size=2)))
    check_consistent(inst, st)


def test_relocate_to_full_school_not_allowed():
    dist = np.array([[1, 2], [2, 1]])
    st = State(dist, [1, 1], [0, 1])
    assert not st.can_relocate(0, 1)
    assert not st.can_relocate(0, 0)


def test_state_rejects_infeasible_start():
    with pytest.raises(ValueError):
        State(np.ones((3, 2)), [1, 2], [0, 0, 1])
