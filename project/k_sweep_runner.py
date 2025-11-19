from project.pursuit import Options, go
from loguru import logger
from project.misc import grid

if __name__ == "__main__":
    loops = 0
    while True:
        for args in grid(
            n=[2**10, 2**14, 2**16, 2**20],
            max_steps=[1],
        ):
            logger.info(f"Completed loops: {loops}")
            go(Options(k_step=1, d_step=64, **args, device="cuda"))
        loops += 1
