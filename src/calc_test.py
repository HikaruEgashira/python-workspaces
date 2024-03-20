from src import calc


def test_add() -> None:
    assert calc.add(1, 2) == 3
