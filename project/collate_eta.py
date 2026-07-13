from pathlib import Path
import pandas as pd

import vandc


def average_results(df):
    experiment_vars = set(["dict_type", "method", "n", "k", "d"])
    displayed_vars = set(["dict_type", "method", "n", "eta", "d_per_nat"])
    all_vars = experiment_vars.union(displayed_vars)

    graph_cells = df[list(all_vars)].drop_duplicates()
    graph_cells = graph_cells.sort_values(list(all_vars))

    results = df.groupby(list(experiment_vars)).mean("acc")
    results = results.reset_index().set_index(list(experiment_vars))[["acc"]]
    return graph_cells.join(results, on=list(experiment_vars), validate="m:1")



def collate_from_path(path: Path):
    runs = list(vandc.fetch_dir(path))
    print(f"read {len(runs)} runs from {path}")
    return vandc.collate_runs(runs)


if __name__ == "__main__":
    # read results from eta_sweep_2, discarding results on top-k and map
    df_gmp = collate_from_path(Path("results") / "eta_sweep_2")
    df_gmp = df_gmp[(df_gmp["method"] != "top-k") & (df_gmp["method"] != "map")]

    # read new results with more runs on just one-step methods
    df = collate_from_path(Path("results") / "eta_sweep_one_step")

    # average success probability per value of (dictionary, n, eta, d_per_nat)
    df = average_results(pd.concat([df, df_gmp]))
    df.to_csv(Path("results") / "eta_sweep_2.csv", index=False)
