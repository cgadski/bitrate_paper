from project.eta_sweep import Options, go
from loguru import logger
from project.misc import grid

if __name__ == "__main__":
    loops = 0
    while True:
        for args in grid(
            n=[2**14, 2**18],
            method=["threshold", 1, 4, 64],
        ):
            logger.info(f"Completed loops: {loops}")
            opts = {
                "n": args["n"],
                "max_factor": 9,
                "max_eta": 0.4,
                "max_floats": 4294967296,
                "device": "cuda",
                "batch": 256,
            }
            if args["method"] == "threshold":
                opts["threshold"] = True
            else:
                opts["max_steps"] = args["method"]

            go(Options(**opts))

        loops += 1
