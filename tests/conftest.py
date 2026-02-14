import pytest
import numsim_cas as cas


@pytest.fixture
def x():
    return cas.variable("x")


@pytest.fixture
def y():
    return cas.variable("y")


@pytest.fixture
def z():
    return cas.variable("z")


@pytest.fixture
def T():
    """Rank-2, dim-3 tensor variable."""
    return cas.tensor_variable("T", dim=3, rank=2)


@pytest.fixture
def S():
    """Rank-2, dim-3 tensor variable."""
    return cas.tensor_variable("S", dim=3, rank=2)
