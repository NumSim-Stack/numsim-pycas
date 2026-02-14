import pytest
import numsim_cas as cas


class TestScalarTensor:
    def test_scalar_times_tensor(self):
        x = cas.variable("x")
        T = cas.tensor_variable("T", dim=3, rank=2)
        result = x * T
        assert isinstance(result, cas.TensorExpr)
        s = str(result)
        assert "x" in s and "T" in s

    def test_tensor_times_scalar(self):
        x = cas.variable("x")
        T = cas.tensor_variable("T", dim=3, rank=2)
        result = T * x
        assert isinstance(result, cas.TensorExpr)

    def test_tensor_div_scalar(self):
        x = cas.variable("x")
        T = cas.tensor_variable("T", dim=3, rank=2)
        result = T / x
        assert isinstance(result, cas.TensorExpr)

    def test_constant_times_tensor(self):
        T = cas.tensor_variable("T", dim=3, rank=2)
        c = cas.constant(2)
        result = c * T
        assert isinstance(result, cas.TensorExpr)

    def test_int_times_tensor(self):
        T = cas.tensor_variable("T", dim=3, rank=2)
        result = 3 * T
        assert isinstance(result, cas.TensorExpr)


class TestScalarT2S:
    def test_scalar_plus_t2s(self):
        x = cas.variable("x")
        T = cas.tensor_variable("T", dim=3, rank=2)
        tr = cas.trace(T)
        result = x + tr
        assert isinstance(result, cas.T2SExpr)

    def test_t2s_plus_scalar(self):
        x = cas.variable("x")
        T = cas.tensor_variable("T", dim=3, rank=2)
        tr = cas.trace(T)
        result = tr + x
        assert isinstance(result, cas.T2SExpr)

    def test_scalar_times_t2s(self):
        x = cas.variable("x")
        T = cas.tensor_variable("T", dim=3, rank=2)
        tr = cas.trace(T)
        result = x * tr
        assert isinstance(result, cas.T2SExpr)

    def test_t2s_times_scalar(self):
        x = cas.variable("x")
        T = cas.tensor_variable("T", dim=3, rank=2)
        tr = cas.trace(T)
        result = tr * x
        assert isinstance(result, cas.T2SExpr)

    def test_t2s_div_scalar(self):
        x = cas.variable("x")
        T = cas.tensor_variable("T", dim=3, rank=2)
        tr = cas.trace(T)
        result = tr / x
        assert isinstance(result, cas.T2SExpr)


class TestMixedExpressions:
    def test_trace_addition(self):
        T = cas.tensor_variable("T", dim=3, rank=2)
        S = cas.tensor_variable("S", dim=3, rank=2)
        result = cas.trace(T) + cas.trace(S)
        s = str(result)
        assert "tr" in s

    def test_constant_times_tensor_string(self):
        T = cas.tensor_variable("T", dim=3, rank=2)
        result = 2 * T
        s = str(result)
        assert "2" in s and "T" in s

    def test_compound_expression(self):
        x = cas.variable("x")
        T = cas.tensor_variable("T", dim=3, rank=2)
        # x * T should give a TensorExpr
        result = x * T
        assert isinstance(result, cas.TensorExpr)
        # trace of that should give a T2SExpr
        tr = cas.trace(result)
        assert isinstance(tr, cas.T2SExpr)
