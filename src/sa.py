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
