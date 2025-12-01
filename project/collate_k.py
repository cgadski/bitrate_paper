# %%
import vandc
from vandc.writer import git_root
from pathlib import Path

root: Path = git_root()  # pyright: ignore
runs = list(vandc.fetch_dir(root / "results" / "k_sweep"))
df = vandc.collate_runs(runs)
df[["n", "k", "d", "acc", "method"]].groupby(["n", "k", "d", "method"]).mean().reset_index().to_csv(root / "results" / "k_sweep.csv", index=False)
