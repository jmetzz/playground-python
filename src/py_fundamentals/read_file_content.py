import sys


def load(filename):
    with open(filename, encoding="utf-8") as f:
        return [int(line.strip()) for line in f]


def cat(filename):
    with open(filename, encoding="utf-8") as f:
        for line in f:
            sys.stdout.write(line)


def main(filename):
    series = load(filename)
    print("Loaded content: ")
    print(series)

    print("------")
    print("cat ->")
    cat(filename)


if __name__ == "__main__":
    main(sys.argv[1])
