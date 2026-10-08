"""Etalons: precīzs algoritms, apakšējā robeža un pilnā pārlase.

Precīzais algoritms ir minimālo izmaksu plūsma (min-cost flow) ar
secīgiem īsākajiem ceļiem (successive shortest paths):

    avots -> skolēns k (caurlaidība 1, izmaksas 0)
    skolēns k -> skola j (caurlaidība 1, izmaksas dist[k, j])
    skola j -> noteka (caurlaidība cap[j], izmaksas 0)

Skolēnus pievieno pa vienam. Ja pirmie t skolēni ir piekārtoti optimāli,
tad īsākais papildinošais ceļš atlikuma grafā no skolēna t+1 līdz skolai
ar brīvu vietu dod optimālu piekārtojumu t+1 skolēniem (tas pats princips,
kas Ungāru algoritmā). Atlikuma grafā ceļš iet pa skolām: no skolas i uz
skolu j var nokļūt, pārceļot kādu skolas i skolēnu k uz j, izmaksas
dist[k, j] - dist[k, i]. Tāpēc īsāko ceļu meklē grafā ar n virsotnēm, kura
šķautnes svars W[i, j] = min_{k skolā i} (dist[k, j] - dist[k, i]).
Svari var būt negatīvi (bet negatīvu ciklu nav, jo pašreizējais
piekārtojums ir optimāls), tāpēc izmanto Belmana-Forda algoritmu
(vektorizētu ar numpy). Pēc katra papildinājuma pārrēķina tikai to skolu
W rindas, kuru skolēnu sastāvs mainījās.

Sarežģītība: O(m * n^3) sliktākajā gadījumā, praksē daudz mazāk.
"""

import itertools

import numpy as np

from .sa import total_cost, validate


def min_cost_assignment(dist, cap):
    """Optimāls piekārtojums (min-cost flow). Atgriež (assign, izmaksas)."""
    dist = np.asarray(dist, dtype=np.int64)
    cap = np.asarray(cap, dtype=np.int64)
    m, n = dist.shape
    if cap.sum() < m:
        raise ValueError("kopējā skolu ietilpība ir mazāka par skolēnu skaitu")
    distf = dist.astype(np.float64)  # veseli skaitļi < 2^53 ir precīzi
    cols = np.arange(n)

    assign = np.full(m, -1, dtype=np.int64)
    load = np.zeros(n, dtype=np.int64)
    members = [[] for _ in range(n)]
    W = np.full((n, n), np.inf)          # skolu grafa šķautņu svari
    A = np.full((n, n), -1, dtype=np.int64)  # kurš skolēns realizē W[i, j]

    def refresh(i):
        if not members[i]:
            W[i] = np.inf
            A[i] = -1
            return
        mem = np.array(members[i])
        sub = distf[mem] - distf[mem, i][:, None]
        idx = sub.argmin(axis=0)
        W[i] = sub[idx, cols]
        A[i] = mem[idx]
        W[i, i] = np.inf

    for k0 in range(m):
        # Belmans-Fords no skolēna k0: D[j] -- īsākā ceļa garums līdz skolai j.
        D = distf[k0].copy()
        pred = np.full(n, -1, dtype=np.int64)
        for _ in range(n):
            cand = D[:, None] + W
            via = cand.argmin(axis=0)
            newD = cand[via, cols]
            better = newD < D
            if not better.any():
                break
            D[better] = newD[better]
            pred[better] = via[better]
        else:
            raise RuntimeError("negatīvs cikls -- nav sagaidāms optimālam stāvoklim")

        D[load >= cap] = np.inf
        j = j_end = int(D.argmin())
        changed = {j}
        steps = 0
        while pred[j] != -1:  # pārceļ skolēnus pa ceļu no beigām uz sākumu
            i = int(pred[j])
            k = int(A[i, j])
            members[i].remove(k)
            members[j].append(k)
            assign[k] = j
            changed.add(i)
            j = i
            steps += 1
            if steps > n:
                raise RuntimeError("cikls priekšteču kokā")
        assign[k0] = j
        members[j].append(k0)
        load[j_end] += 1
        for i in changed:
            refresh(i)

    assert validate(dist, cap, assign) is None
    return assign, total_cost(dist, assign)


def lower_bound(dist, cap=None):
    """Apakšējā robeža: katrs skolēns tuvākajā skolā, ignorējot ietilpības."""
    return np.asarray(dist).min(axis=1).sum().item()


def brute_force(dist, cap):
    """Pilnā pārlase pa visiem n^m piešķīrumiem (tikai ļoti maziem piemēriem)."""
    dist = np.asarray(dist)
    m, n = dist.shape
    if n ** m > 2_000_000:
        raise ValueError("piemērs ir par lielu pilnajai pārlasei")
    best_cost, best = None, None
    for assign in itertools.product(range(n), repeat=m):
        if validate(dist, cap, np.array(assign)) is not None:
            continue
        c = total_cost(dist, assign)
        if best_cost is None or c < best_cost:
            best_cost, best = c, np.array(assign)
    return best, best_cost
