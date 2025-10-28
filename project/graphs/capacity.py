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
        ax.set_ylabel("$C^{-1}$", rotation="horizontal")

        fig.tight_layout()


# %%
if __name__ == "__main__":
    setup()
    CapacityGraph().plot()
    plt.savefig("figures/capacity.pdf", dpi=300)
