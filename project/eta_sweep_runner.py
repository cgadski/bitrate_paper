import argparse

from loguru import logger

from project.eta_sweep import Options, go
from project.misc import grid

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--one-step-only", action="store_true")
    args = parser.parse_args()

    max_floats = 4_294_967_296
    n_range = [2**8, 2**12, 2**16, 2**20]

    if args.small:
        max_floats = 1_000_000
        n_range = n_range[:3]

    loops = 0
    while True:
        for params in grid(
            dict_type=["spherical", "rademacher"],
            n=n_range,
        ):
            logger.info(f"Completed loops: {loops}")
            opts = Options(
                n=params["n"],
                dict_type=params["dict_type"],
                max_d_per_nat=9,
                max_eta=0.4,
                one_step_only=args.one_step_only,
                max_floats=max_floats,
                device="cuda",
                batch=32 if args.small else 256,
            )

            if args.small:
                opts.device = "cpu"

            go(opts)

        loops += 1
