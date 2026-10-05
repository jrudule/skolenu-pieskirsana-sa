"""Skolēnu piekārtošana skolām ar simulēto rūdīšanu.

Apzīmējumi:
    m          -- skolēnu skaits, n -- skolu skaits;
    dist[k, j] -- attālums no skolēna k līdz skolai j (m x n matrica);
    cap[j]     -- skolas j ietilpība;
    assign[k]  -- skola, kurai piešķirts skolēns k (risinājuma domēns).
"""

import numpy as np


# ---------------------------------------------------------------------------
# Novērtēšana un derīguma pārbaude
# ---------------------------------------------------------------------------

def total_cost(dist, assign):
    """Mērķa funkcija: visu skolēnu attālumu summa līdz piešķirtajām skolām."""
    dist = np.asarray(dist)
    assign = np.asarray(assign, dtype=np.int64)
    return dist[np.arange(len(assign)), assign].sum().item()


def school_loads(assign, n):
    """Skolu aizpildījuma skaitītāji: cik skolēnu piešķirts katrai skolai."""
    return np.bincount(np.asarray(assign, dtype=np.int64), minlength=n)


def validate(dist, cap, assign):
    """Atgriež kļūdas aprakstu vai None, ja risinājums ir derīgs.

    Derīgs risinājums: katram skolēnam piešķirta tieši viena eksistējoša skola,
    un nevienā skolā skolēnu nav vairāk par ietilpību.
    """
    dist = np.asarray(dist)
    cap = np.asarray(cap)
    m, n = dist.shape
    assign = np.asarray(assign)
    if assign.shape != (m,):
        return f"piešķīrumu garums {assign.shape} neatbilst skolēnu skaitam {m}"
    if not np.issubdtype(assign.dtype, np.integer):
        return "skolu indeksiem jābūt veseliem skaitļiem"
    if m and (assign.min() < 0 or assign.max() >= n):
        return "piešķirta neeksistējoša skola"
    loads = school_loads(assign, n)
    over = np.flatnonzero(loads > cap)
    if over.size:
        j = over[0]
        return f"skola {j} pārpildīta: {loads[j]} > {cap[j]}"
    return None


def is_feasible(dist, cap, assign):
    """True, ja risinājums ir derīgs (sk. `validate`)."""
    return validate(dist, cap, assign) is None


def evaluate(dist, cap, assign):
    """Novērtē risinājumu; pārpildīti (nederīgi) risinājumi netiek pieļauti."""
    err = validate(dist, cap, assign)
    if err is not None:
        raise ValueError(f"nederīgs risinājums: {err}")
    return total_cost(dist, assign)


# ---------------------------------------------------------------------------
# Sākuma risinājums
# ---------------------------------------------------------------------------

def greedy_initial(dist, cap, order=None):
    """Alkatīgs sākuma risinājums: katram skolēnam tuvākā skola ar brīvu vietu.

    Skolēni tiek apstrādāti secībā `order` (pēc noklusējuma 0, 1, ..., m-1).
    Tā kā kopējā ietilpība >= m, brīva vieta vienmēr atradīsies.
    """
    dist = np.asarray(dist)
    cap = np.asarray(cap)
    m, n = dist.shape
    if cap.sum() < m:
        raise ValueError("kopējā skolu ietilpība ir mazāka par skolēnu skaitu")
    if order is None:
        order = range(m)
    by_distance = np.argsort(dist, axis=1, kind="stable")
    free = cap.astype(np.int64).copy()
    assign = np.full(m, -1, dtype=np.int64)
    for k in order:
        for j in by_distance[k]:
            if free[j] > 0:
                assign[k] = j
                free[j] -= 1
                break
    return assign


# ---------------------------------------------------------------------------
# Stāvoklis un gājieni
# ---------------------------------------------------------------------------

class State:
    """Pašreizējais risinājums ar inkrementāli uzturamiem lielumiem.

    assign[k]  -- skolēna k skola;
    load[j]    -- skolas j aizpildījuma skaitītājs;
    members[j] -- skolas j skolēnu saraksts, pos[k] -- k indekss tajā
                  (vajadzīgs, lai O(1) laikā izvēlētos nejaušu skolas skolēnu);
    cost       -- mērķa funkcijas vērtība.

    Iekšējā ciklā izmanto Python sarakstus, jo atsevišķu elementu piekļuve
    tiem ir ievērojami ātrāka nekā numpy masīviem.
    """

    def __init__(self, dist, cap, assign):
        err = validate(dist, cap, assign)
        if err is not None:
            raise ValueError(f"nederīgs sākuma risinājums: {err}")
        self.m, self.n = np.shape(dist)
        self.dist = np.asarray(dist).tolist()
        self.cap = np.asarray(cap).tolist()
        self.assign = np.asarray(assign).tolist()
        self.load = [0] * self.n
        self.members = [[] for _ in range(self.n)]
        self.pos = [0] * self.m
        for k, j in enumerate(self.assign):
            self.load[j] += 1
            self.pos[k] = len(self.members[j])
            self.members[j].append(k)
        self.cost = total_cost(dist, assign)

    # -- skolēna pārcelšana uz skolu ar brīvu vietu --------------------------

    def can_relocate(self, k, j):
        return j != self.assign[k] and self.load[j] < self.cap[j]

    def delta_relocate(self, k, j):
        """Δ, pārceļot skolēnu k uz skolu j: O(1)."""
        row = self.dist[k]
        return row[j] - row[self.assign[k]]

    def relocate(self, k, j, delta=None):
        if delta is None:
            delta = self.delta_relocate(k, j)
        s = self.assign[k]
        self._remove(k, s)
        self._add(k, j)
        self.assign[k] = j
        self.load[s] -= 1
        self.load[j] += 1
        self.cost += delta

    # -- divu skolēnu apmaiņa (aizpildījums nemainās) ------------------------

    def delta_swap(self, k1, k2):
        """Δ, apmainot skolēnu k1 un k2 skolas: O(1)."""
        s1, s2 = self.assign[k1], self.assign[k2]
        r1, r2 = self.dist[k1], self.dist[k2]
        return r1[s2] + r2[s1] - r1[s1] - r2[s2]

    def swap(self, k1, k2, delta=None):
        if delta is None:
            delta = self.delta_swap(k1, k2)
        s1, s2 = self.assign[k1], self.assign[k2]
        # Saraksta vietas apmainās tieši tāpat kā skolas.
        p1, p2 = self.pos[k1], self.pos[k2]
        self.members[s1][p1] = k2
        self.members[s2][p2] = k1
        self.pos[k1], self.pos[k2] = p2, p1
        self.assign[k1], self.assign[k2] = s2, s1
        self.cost += delta

    # -- palīgfunkcijas -----------------------------------------------------

    def _remove(self, k, j):
        lst = self.members[j]
        last = lst.pop()
        if last != k:
            p = self.pos[k]
            lst[p] = last
            self.pos[last] = p

    def _add(self, k, j):
        self.pos[k] = len(self.members[j])
        self.members[j].append(k)

    def assignment(self):
        return np.array(self.assign, dtype=np.int64)
