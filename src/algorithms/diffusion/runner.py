from timeit import timeit

import numpy as np
from algorithms.diffusion.diffusion_np import (
    DiffusionNumpy,
    DiffusionNumpyInPlace,
    DiffusionNumpyInPlaceLessAllocation,
)
from algorithms.diffusion.diffusion_numexpr import DiffusionNumexpr


class Runner:
    @staticmethod
    def run_experiment(diffuser, num_iterations, dt=0.1, low_factor=0.4, high_factor=0.5):
        next_grid = np.zeros(diffuser.grid_shape)
        grid = np.zeros(diffuser.grid_shape)

        block_low = int(diffuser.grid_shape[0] * low_factor)
        block_high = int(diffuser.grid_shape[0] * high_factor)
        grid[block_low:block_high, block_low:block_high] = 0.005

        for _ in range(num_iterations):
            diffuser.evolve(grid, dt, next_grid)
            grid, next_grid = next_grid, grid
        return grid


def measure_time(diffuser, num_iterations=500, num_executions=10):
    return timeit(
        lambda: Runner.run_experiment(diffuser, num_iterations),
        number=num_executions,
    )


def _test_roll_add(clz):
    rollee = np.asarray([[1, 2], [3, 4]])
    for shift in (-1, +1):
        for axis in (0, 1):
            out = np.asarray([[6, 3], [9, 2]])
            expected_result = np.roll(rollee, shift, axis=axis) + out
            clz.roll_add(rollee, shift, axis, out)
            if not np.all(expected_result == out):
                raise ValueError(f"roll_add not working for class {clz}")


if __name__ == "__main__":
    _test_roll_add(DiffusionNumpyInPlaceLessAllocation)
    _test_roll_add(DiffusionNumexpr)

    grid_shape = (640, 640)

    print(measure_time(DiffusionNumpy(grid_shape)))
    print(measure_time(DiffusionNumpyInPlace(grid_shape)))
    print(measure_time(DiffusionNumpyInPlaceLessAllocation(grid_shape)))
    print(measure_time(DiffusionNumexpr(grid_shape)))
