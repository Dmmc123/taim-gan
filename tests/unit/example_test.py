from src.test_project import Foo


def test_dummy() -> None:
    f1 = Foo(1, 2)
    f2 = Foo(2, 1)
    assert f1.get_bar() == f2.get_foo(), "2 = 2 * 1"
