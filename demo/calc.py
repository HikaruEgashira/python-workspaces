def add(a: int, b: int) -> int:
    return a + b


def test_add() -> None:
    assert add(1, 2) == 3


if __name__ == "__main__":
    print(f"{add(1, 2)=}")
