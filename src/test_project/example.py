"""doing some stuff here"""


class Foo:
    """sample text"""

    def __init__(self, first_var: int, second_var: int) -> None:
        """init the bar"""
        self.first = first_var
        self.second = second_var

    def get_bar(self) -> int:
        """return bar"""
        return self.first

    def get_foo(self) -> int:
        """return bar"""
        return self.second
