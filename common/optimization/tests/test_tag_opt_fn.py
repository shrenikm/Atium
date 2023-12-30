import pytest

from common.optimization.tag_opt_fn import is_tagged_opt_fn, tag_atium_opt_fn


def test_opt_fn_tagging():
    def f1(x: float) -> float:
        return x

    assert not is_tagged_opt_fn(fn=f1)

    @tag_atium_opt_fn
    def f2(x: float) -> float:
        return x

    assert is_tagged_opt_fn(fn=f2)

    @tag_atium_opt_fn(use_jit=False)
    def f3(x: float) -> float:
        return x

    assert is_tagged_opt_fn(fn=f3)


def test_opt_fn_tag_scalar_input_scalar_output():
    ...


if __name__ == "__main__":

    pytest.main(["-s", "-v", __file__])
