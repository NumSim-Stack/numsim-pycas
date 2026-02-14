import pytest
import numsim_cas as cas


class TestT2SConstruction:
    def test_trace(self, T):
        result = cas.trace(T)
        assert "tr" in str(result)

    def test_det(self, T):
        result = cas.det(T)
        assert "det" in str(result)

    def test_norm(self, T):
        result = cas.norm(T)
        s = str(result)
        assert "norm" in s or "||" in s

    def test_dot(self, T):
        result = cas.dot(T)
        assert isinstance(result, cas.T2SExpr)


class TestT2SOperators:
    def test_add(self, T, S):
        a = cas.trace(T)
        b = cas.trace(S)
        result = a + b
        s = str(result)
        assert "tr" in s

    def test_sub(self, T, S):
        a = cas.trace(T)
        b = cas.trace(S)
        result = a - b
        assert isinstance(result, cas.T2SExpr)

    def test_mul(self, T, S):
        a = cas.trace(T)
        b = cas.trace(S)
        result = a * b
        assert isinstance(result, cas.T2SExpr)

    def test_div(self, T, S):
        a = cas.trace(T)
        b = cas.trace(S)
        result = a / b
        assert isinstance(result, cas.T2SExpr)

    def test_neg(self, T):
        a = cas.trace(T)
        result = -a
        assert isinstance(result, cas.T2SExpr)

    def test_pow_int(self, T):
        a = cas.trace(T)
        result = a ** 2
        assert isinstance(result, cas.T2SExpr)


class TestT2SWithNumeric:
    def test_add_int(self, T):
        a = cas.trace(T)
        result = a + 1
        assert isinstance(result, cas.T2SExpr)

    def test_radd_int(self, T):
        a = cas.trace(T)
        result = 1 + a
        assert isinstance(result, cas.T2SExpr)

    def test_mul_int(self, T):
        a = cas.trace(T)
        result = a * 2
        assert isinstance(result, cas.T2SExpr)

    def test_rmul_int(self, T):
        a = cas.trace(T)
        result = 2 * a
        assert isinstance(result, cas.T2SExpr)


class TestT2SFunctions:
    def test_log(self, T):
        a = cas.trace(T)
        result = cas.log(a)
        assert isinstance(result, cas.T2SExpr)

    def test_pow_function(self, T):
        a = cas.trace(T)
        result = cas.pow(a, 2)
        assert isinstance(result, cas.T2SExpr)


class TestT2SComparison:
    def test_equality(self):
        T = cas.tensor_variable("T", dim=3, rank=2)
        a = cas.trace(T)
        b = cas.trace(T)
        assert a == b

    def test_hash(self, T):
        a = cas.trace(T)
        h = hash(a)
        assert isinstance(h, int)
