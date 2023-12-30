import pytest

from common.optimization.kernel import atium_opt_fn_kernel, is_tagged_opt_fn


def test_opt_fn_tagging():
    def f1(x: float) -> float:
        return x

    assert not is_tagged_opt_fn(fn=f1)

    @atium_opt_fn_kernel
    def f2(x: float) -> float:
        return x

    assert is_tagged_opt_fn(fn=f2)

    @atium_opt_fn_kernel(use_jit=False)
    def f3(x: float) -> float:
        return x

    assert is_tagged_opt_fn(fn=f3)


def test_opt_fn_tag_scalar_input_scalar_output():
    ...


if __name__ == "__main__":

    pytest.main(["-s", "-v", __file__])
