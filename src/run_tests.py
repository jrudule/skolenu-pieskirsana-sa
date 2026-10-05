"""Eksperimenti: simulētā rūdīšana pret etalonu uz standarta piemēriem.

Palaišana (no repozitorija saknes):
    python -m src.run_tests                 # pilns eksperiments (~1-2 min)
    python -m src.run_tests --runs 3 --no-alpha   # ātrāka pārbaude

Rezultāti:
    results/results.csv      -- kopsavilkums pa piemēriem
    results/runs.csv         -- katra palaišana atsevišķi
    results/alpha_sweep.csv  -- dzesēšanas koeficienta alfa ietekme
    results/*.png            -- grafiki
"""

import argparse
import csv
import os
import time

import numpy as np

from .exact import lower_bound, min_cost_assignment
from .generator import standard_instances
from .sa import greedy_initial, is_feasible, simulated_annealing, total_cost

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")


def gap(value, ref):
    """Novirze % no etalona."""
    return 100.0 * (value - ref) / ref


def reference(inst, exact_max_m):
    """Etalons: precīzs optimums, ja m <= exact_max_m, citādi apakšējā robeža."""
    lb = lower_bound(inst.dist)
    if inst.m <= exact_max_m:
        t = time.perf_counter()
        assign, opt = min_cost_assignment(inst.dist, inst.cap)
        t_exact = time.perf_counter() - t
        assert is_feasible(inst.dist, inst.cap, assign)
        return dict(ref=opt, ref_type="optimums", opt=opt, lb=lb, exact_time=t_exact)
    return dict(ref=lb, ref_type="apakšējā robeža", opt=None, lb=lb, exact_time=None)


def run_instance(inst, runs, exact_max_m):
    ref = reference(inst, exact_max_m)
    greedy = total_cost(inst.dist, greedy_initial(inst.dist, inst.cap))
    rows, histories = [], {}
    for seed in range(runs):
        t = time.perf_counter()
        res = simulated_annealing(inst.dist, inst.cap, seed=seed)
        elapsed = time.perf_counter() - t
        assert is_feasible(inst.dist, inst.cap, res.assign)
        rows.append(dict(instance=inst.name, m=inst.m, n=inst.n, seed=seed, cost=res.cost,
                         gap_pct=gap(res.cost, ref["ref"]), time_s=elapsed,
                         iterations=res.iterations, accepted=res.accepted, t0=res.t0))
        histories[seed] = res.history
    costs = np.array([r["cost"] for r in rows])
    times = np.array([r["time_s"] for r in rows])
    summary = dict(
        instance=inst.name, m=inst.m, n=inst.n, capacity=int(inst.cap.sum()),
        reference=ref["ref_type"], ref=ref["ref"], optimum=ref["opt"], lower_bound=ref["lb"],
        greedy=greedy, greedy_gap_pct=gap(greedy, ref["ref"]),
        sa_best=int(costs.min()), sa_mean=float(costs.mean()), sa_std=float(costs.std(ddof=1)) if runs > 1 else 0.0,
        sa_worst=int(costs.max()),
        best_gap_pct=gap(costs.min(), ref["ref"]), mean_gap_pct=gap(costs.mean(), ref["ref"]),
        worst_gap_pct=gap(costs.max(), ref["ref"]),
        hits_optimum=int((costs == ref["opt"]).sum()) if ref["opt"] is not None else None,
        sa_time_mean_s=float(times.mean()), sa_time_std_s=float(times.std(ddof=1)) if runs > 1 else 0.0,
        exact_time_s=ref["exact_time"],
        opt_vs_lb_pct=gap(ref["opt"], ref["lb"]) if ref["opt"] is not None else None,
    )
    return summary, rows, histories


def alpha_sweep(inst, alphas, seeds, opt):
    rows = []
    for a in alphas:
        for seed in range(seeds):
            t = time.perf_counter()
            res = simulated_annealing(inst.dist, inst.cap, seed=seed, alpha=a)
            rows.append(dict(instance=inst.name, alpha=a, seed=seed, cost=res.cost,
                             gap_pct=gap(res.cost, opt), time_s=time.perf_counter() - t,
                             iterations=res.iterations))
    return rows


def fmt(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.4f}"
    return v


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: fmt(v) for k, v in r.items()})


def print_table(summaries):
    head = f"{'piemērs':<12}{'etalons':>9}{'labākais':>10}{'vidējais':>11}{'σ':>8}" \
           f"{'lab.%':>8}{'vid.%':>8}{'alk.%':>8}{'laiks,s':>9}{'prec.,s':>9}"
    print(head)
    print("-" * len(head))
    for s in summaries:
        ex = f"{s['exact_time_s']:.2f}" if s["exact_time_s"] is not None else "-"
        print(f"{s['instance']:<12}{s['ref']:>9}{s['sa_best']:>10}{s['sa_mean']:>11.1f}{s['sa_std']:>8.1f}"
              f"{s['best_gap_pct']:>8.2f}{s['mean_gap_pct']:>8.2f}{s['greedy_gap_pct']:>8.2f}"
              f"{s['sa_time_mean_s']:>9.2f}{ex:>9}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", type=int, default=10, help="palaišanu skaits katram piemēram")
    p.add_argument("--exact-max-m", type=int, default=2000,
                   help="lielākais m, kam rēķina precīzo optimumu; lielākiem -- apakšējā robeža")
    p.add_argument("--no-alpha", action="store_true", help="izlaist alfa ietekmes eksperimentu")
    p.add_argument("--no-plots", action="store_true", help="nezīmēt grafikus")
    p.add_argument("--out", default=RESULTS_DIR)
    args = p.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    summaries, all_rows, histories = [], [], {}
    for inst in standard_instances():
        print(f"{inst.name}: {args.runs} palaišanas ...", flush=True)
        s, rows, hist = run_instance(inst, args.runs, args.exact_max_m)
        summaries.append(s)
        all_rows += rows
        histories[inst.name] = (s["ref"], hist[0])
    write_csv(os.path.join(args.out, "results.csv"), summaries)
    write_csv(os.path.join(args.out, "runs.csv"), all_rows)
    print()
    print_table(summaries)

    sweep = None
    if not args.no_alpha:
        inst = standard_instances()[3]
        opt = next(s["ref"] for s in summaries if s["instance"] == inst.name)
        print(f"\nalfa ietekme uz {inst.name} ...", flush=True)
        sweep = alpha_sweep(inst, [0.90, 0.95, 0.98, 0.99], 5, opt)
        write_csv(os.path.join(args.out, "alpha_sweep.csv"), sweep)
        for a in sorted({r["alpha"] for r in sweep}):
            g = [r["gap_pct"] for r in sweep if r["alpha"] == a]
            t = [r["time_s"] for r in sweep if r["alpha"] == a]
            print(f"  alfa={a:.2f}: vid. novirze {np.mean(g):.2f}%, vid. laiks {np.mean(t):.2f} s")

    if not args.no_plots:
        from .plots import make_plots
        for path in make_plots(summaries, all_rows, histories, sweep, args.out):
            print("grafiks:", os.path.relpath(path, ROOT))


if __name__ == "__main__":
    main()
