# %%
import pandas as pd
from matplotlib.colors import Normalize
from project.graphs.settings import setup, FIG_WIDTH
import seaborn as sns
import numpy as np
from math import log2
import matplotlib.pyplot as plt


class StorylineGraph:
    def plot(self):
        fig, ax = plt.subplots()
        fig.set_size_inches(FIG_WIDTH, FIG_WIDTH * 0.8)
        k = np.linspace(1, 110, 200)
        n = 1 << 20
        ax.set_ylim(0, 2**13)
        ax.set_xlim(0, 100)
        ax.set_yticks(2 ** np.arange(9, 14))

        ax2 = ax.twinx()
        ax2.set_ylim(0, 2**13)  # same y-range as left axis
        ax2.set_ylabel("Dimensions per bit")

        one_bit = 100 * (np.log(n / 100) + 1) / np.log(2)
        ax2.set_yticks(np.arange(1, 6) * one_bit)
        ax2.set_yticklabels(np.arange(1, 6))

        ax.plot(k, 4 * k * np.log(n * k))
        k_label = 65
        ax.annotate(
            "Top-k",
            xy=(k_label, 4 * k_label * np.log(n * k_label)),
            xytext=(-23, 4),
            textcoords="offset points",
        )

        ax.plot(k, 1.3 * k * (1 + np.log(n / k)) / np.log(2))
        k_label = 70
        ax.annotate(
            "Matching pursuit",
            xy=(k_label, 1.3 * k_label * (1 + np.log(n / k_label)) / np.log(2)),
            xytext=(-20, 12),
            textcoords="offset points",
        )

        ax.set_xlabel("$k$")
        ax.set_ylabel("$d$")

        for b in range(1, 7):
            ax.plot(
                k, k * b * (np.log(n / k) + 1) / np.log(2), "--", lw=0.5, color="grey"
            )

        fig.tight_layout()


# %%
if __name__ == "__main__":
    setup()
    StorylineGraph().plot()
    plt.savefig("figures/storyline.pdf", dpi=300)
