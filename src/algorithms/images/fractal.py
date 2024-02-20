"""Computing Mandelbrot sets."""

import math

import matplotlib.pyplot as plt


def mandel(real, imag):
    """
    Compute a point in the Mandelbrot.

    The logarithm of number of iterations needed to determine
    whether a complex point is in the Mandelbrot set.

    Args:
        real: The real coordinate
        imag: the imaginary coordinate

    Returns:
        An integer in the range 1-255
    """
    x = 0
    y = 0
    i = 1
    while i < 257:
        if x * x + y * y > 4.0:
            break
        xt = x * x - y * y + real
        y = 2.0 * x * y + imag
        x = xt
        i += 1
    else:
        # If the loop completes normally,
        # use the maximum iterations to calculate the color
        return 255
    # Adjust the return statement to handle edge cases
    # and ensure output is within the desired range
    if i == 1:
        # Return 0 if the point diverges immediately (to avoid math domain error)
        return 0
    return int(math.log(i) * 256 / math.log(256)) - 1


def mandelbrot(size_x, size_y):
    """
    Make a Mandelbrot set image

    Args:
        size_x: Image width
        size_y: Image height

    Returns:
        A list of lists of integers in the range 0-255
    """
    return [[mandel((3.5 * x / size_x) - 2.5, (2.0 * y / size_y) - 1.0) for x in range(size_x)] for y in range(size_y)]


if __name__ == "__main__":
    # Define the size of the image
    size_x, size_y = 800, 600

    # Generate the Mandelbrot set data
    data = mandelbrot(size_x, size_y)

    # Create a figure and axis for the plot
    fig, ax = plt.subplots()

    # Display the data
    ax.imshow(data, origin="lower", extent=(-2.5, 1.0, -1.0, 1.0), cmap="hot")

    # Set title and labels (optional)
    ax.set_title("Mandelbrot Set")
    ax.set_xlabel("Real axis")
    ax.set_ylabel("Imaginary axis")

    # Show the plot
    plt.show()
