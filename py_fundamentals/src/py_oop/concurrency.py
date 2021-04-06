class Base:
    counter = 0


class X(Base):
    def inc(self):
        self.counter = self.counter + 1

    def dec(self):
        self.counter = self.counter - 1


if __name__ == "__main__":
    y = type("Y", (X,), dict())()
    z = type("Z", (X,), dict())()

    y.inc()
    print(y.counter)
    z.inc()
    z.inc()
    print(z.counter)
    print(y.counter)
