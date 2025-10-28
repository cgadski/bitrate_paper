import matplotlib.pyplot as plt


def setup():
    plt.rcParams["text.usetex"] = True
    font = {
        "family": "normal",
        "size": 8,
    }

    plt.rc("font", **font)


FIG_WIDTH = 3.25
C_HUE = 220
