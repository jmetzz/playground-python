# headlines.py


def headline(text: str, center: bool = True) -> str:
    if center:
        return f"{text.title()}\n{'-' * len(text)}"
    else:
        return f" {text.title()} ".center(50, "o")


print(headline("python type checking"))
print(headline("use mypy"))
