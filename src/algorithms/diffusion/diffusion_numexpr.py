"""Diffusion is one of the mechanisms that moves fluids and
tries to make them uniformly mixed. The diffusion equation is
also called the heat equation.

Understanding the motivation behind your code and the intricacies
of the algorithm will give you deeper insight about possible methods
of optimization [Gorelick & Ozsvald].

What we will do is take the diffusion equation, which is continuous
in space and time, and approximate it using discrete volumes and
discrete times. We will do so using Euler’s method.
Euler’s method simply takes the derivative and writes it as a difference.

See [Wikipedia page on the diffusion](https://en.wikipedia.org/wiki/Diffusion_equation)
equation and Chapter 7 of Numerical Methods for Complex Systems
by S. V. Gurevich, for details.
"""

import numexpr
import numpy as np


class DiffusionNumexpr:
    """This version uses numexpr to further optimize large matrix operations"""

    def __init__(self, grid_shape):
        self._grid_shape = grid_shape

    @classmethod
    def laplacian(cls, grid, out):
        np.copyto(out, grid)
        out *= -4
        cls.roll_add(grid, +1, 0, out)
        cls.roll_add(grid, -1, 0, out)
        cls.roll_add(grid, +1, 1, out)
        cls.roll_add(grid, -1, 1, out)

    @classmethod
    def roll_add(cls, rollee, shift, axis, out):
        """Given a matrix, a rollee, and an output matrix, out, this function will
        perform the calculation:

            >>> out += np.roll(rollee, shift, axis=axis)

        This is done with the following assumptions:
            * rollee is 2D
            * shift will only ever be +1 or -1
            * axis will only ever be 0 or 1 (also implied by the first assumption)

        Using these assumptions, we are able to speed up this function by avoiding
        extra machinery that numpy uses to generalize the roll function and also
        by making this operation intrinsically in-place.
        """
        if shift == 1 and axis == 0:
            out[1:, :] += rollee[:-1, :]
            out[0, :] += rollee[-1, :]
        elif shift == -1 and axis == 0:
            out[:-1, :] += rollee[1:, :]
            out[-1, :] += rollee[0, :]
        elif shift == 1 and axis == 1:
            out[:, 1:] += rollee[:, :-1]
            out[:, 0] += rollee[:, -1]
        elif shift == -1 and axis == 1:
            out[:, :-1] += rollee[:, 1:]
            out[:, -1] += rollee[:, 0]

    @classmethod
    def evolve(cls, grid, dt, next_grid, D=1):
        """This version uses the out parameter
        of the evaluate function so that
        numexpr doesn’t allocate a new vector
        to which to return the result
        of the calculation
        """
        cls.laplacian(grid, next_grid)
        numexpr.evaluate("next_grid * D * dt + grid", out=next_grid)
