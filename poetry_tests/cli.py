#!/usr/bin/env python3

import click

from src.primes import prime_check


@click.group()
def main():
    pass


@main.command()
@click.option("-n", type=click.INT, default=10_000_000)
def check(n):
    print(f"check_prime({n}) = {prime_check(n)}")


if __name__ == "__main__":
    main()
