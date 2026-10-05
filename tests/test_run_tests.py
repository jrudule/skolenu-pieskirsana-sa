import csv

from src.run_tests import main


def test_run_tests_smoke(tmp_path):
    main(["--runs", "2", "--no-alpha", "--no-plots", "--out", str(tmp_path)])
    with open(tmp_path / "results.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 5
    for r in rows:
        assert float(r["lower_bound"]) <= float(r["optimum"]) <= float(r["sa_best"])
        assert float(r["best_gap_pct"]) >= 0
    with open(tmp_path / "runs.csv", encoding="utf-8") as f:
        assert len(list(csv.DictReader(f))) == 10
