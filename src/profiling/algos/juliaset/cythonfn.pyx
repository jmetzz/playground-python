#cython: boundscheck=False

# cythonfn.pyx
def calculate_z(maxiter, zs, cs):
    """Calculate output list using Julia update rule

    $ make clean compile
    $ make time_cy
    >>> Measuring time for cython juliaset ...
    poetry run /usr/bin/time -l python -m juliaset_cy
    Length of x: 1000
    Total elements: 1000000
    Took 3.24 seconds
            3.73 real         3.66 user         0.06 sys
            116498432  maximum resident set size
                    0  average shared memory size
                    0  average unshared data size
                    0  average unshared stack size
                33967  page reclaims
                    0  page faults
                    0  swaps
                    0  block input operations
                    0  block output operations
                    0  messages sent
                    0  messages received
                    0  signals received
                    0  voluntary context switches
                    568  involuntary context switches
            42541803436  instructions retired
            14216700032  cycles elapsed
            114094080  peak memory footprint
    """
    output = [0] * len(zs)
    for i in range(len(zs)):
        n = 0
        z = zs[i]
        c = cs[i]
        while n < maxiter and abs(z) < 2:
            z = z * z + c
            n += 1
        output[i] = n
    return output


def calculate_z_typed(int maxiter, zs, cs):
    """Calculate output list using Julia update rule

    When adding Cython annotations, you’re adding non-Python code to the .pyx file. This means you lose the interactive nature of developing Python in the interpreter.

    It is important to note that these types will be understood only by Cython and not by Python. Cython uses these types to convert the Python code to C objects, which do not have to call back into the Python stack; this means the operations run at a faster speed, but they lose flexibility and development speed.

    $ make clean compile
    $ make time_cy
    >>> Measuring time for cython juliaset ...
    poetry run /usr/bin/time -l python -m juliaset_cy
    Length of x: 1000
    Total elements: 1000000
    Took 0.21 seconds
            0.73 real         0.66 user         0.05 sys
            120098816  maximum resident set size
                    0  average shared memory size
                    0  average unshared data size
                    0  average unshared stack size
                33003  page reclaims
                    0  page faults
                    0  swaps
                    0  block input operations
                    0  block output operations
                    0  messages sent
                    0  messages received
                    0  signals received
                    12  voluntary context switches
                    84  involuntary context switches
            5966742117  instructions retired
            2628950028  cycles elapsed
            115752960  peak memory footprint

    """
    cdef unsigned int i, n
    cdef double complex z, c
    output = [0] * len(zs)
    for i in range(len(zs)):
        n = 0
        z = zs[i]
        c = cs[i]
        while n < maxiter and abs(z) < 2:
            z = z * z + c
            n += 1
        output[i] = n
    return output

def calculate_z_simple_condition(int maxiter, zs, cs):
    """Calculate output list using Julia update rule

    Replace the relatively expensive abs function with a simplified line of expanded mathematics.

    $ make clean compile
    $ make time_cy
    >>> Measuring time for cython juliaset ...
    poetry run /usr/bin/time -l python -m juliaset_cy
    Length of x: 1000
    Total elements: 1000000
    Took 0.15 seconds
            0.66 real         0.60 user         0.05 sys
            118779904  maximum resident set size
                    0  average shared memory size
                    0  average unshared data size
                    0  average unshared stack size
                32824  page reclaims
                    0  page faults
                    0  swaps
                    0  block input operations
                    0  block output operations
                    0  messages sent
                    0  messages received
                    0  signals received
                    6  voluntary context switches
                    92  involuntary context switches
            5109506567  instructions retired
            2398039122  cycles elapsed
            116314112  peak memory footprint
    """
    cdef unsigned int i, n
    cdef double complex z, c
    output = [0] * len(zs)
    for i in range(len(zs)):
        n = 0
        z = zs[i]
        c = cs[i]
        while n < maxiter and (z.real * z.real + z.imag * z.imag) < 4:
            z = z * z + c
            n += 1
        output[i] = n
    return output