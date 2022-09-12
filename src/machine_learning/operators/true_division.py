class InfixTrueDivision(object):
    def __init__(self, func):
        self.func = func

    class RBind:
        def __init__(self, func, binded):
            self.func = func
            self.binded = binded
        def __call__(self, other):
            return self.func(other, self.binded)
        __rtruediv__ = __call__

    class LBind:
        def __init__(self, func, binded):
            self.func = func
            self.binded = binded
        def __call__(self, other):
            return self.func(self.binded, other)
        __truediv__ = __call__

    def __truediv__(self, other):
        return self.RBind(self.func, other)

    def __rtruediv__(self, other):
        return self.LBind(self.func, other)

    def __call__(self, value1, value2):
        return self.func(value1, value2)

@InfixTrueDivision
def allow_zero(a: float, b: float):
    return a / b if b else 0