# %%
from project.graphs.settings import setup, FIG_WIDTH
import numpy as np
import matplotlib.pyplot as plt


class CapacityGraph:
    def plot(self):
        fig, ax = plt.subplots()
        fig.set_size_inches(FIG_WIDTH, FIG_WIDTH * 0.8)

        eta = np.linspace(0, 0.999, 500)
        c = (2 + 4 * np.sqrt(eta) + 2 * eta) / (1 - eta)
        ax.set_ylim(0, 12)
        ax.set_xlim(0, 0.5)

        ax.plot(eta, c)

        ax.set_xlabel("$\\eta$")
        ax.set_ylabel("dims. per nat")

        ax2 = ax.twinx()
        ax2.set_ylim(0, 12 * np.log(2))
        ax2.set_ylabel("dims. per bit")
        ax2.set_yticks(range(0, int(12 * np.log(2)) + 1, 2))

        ax.grid(True)

        fig.tight_layout()


# %%
if __name__ == "__main__":
    setup()
    CapacityGraph().plot()
    plt.savefig("figures/capacity.pdf", dpi=300)
