from pathlib import Path

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


if __name__ == "__main__":
    runs = list(vandc.fetch_dir(Path("results") / "eta_sweep_2"))
    print(f"Found {len(runs)} runs")
    df = average_results(vandc.collate_runs(runs))
    df.to_csv(Path("results") / "eta_sweep_2.csv", index=False)
