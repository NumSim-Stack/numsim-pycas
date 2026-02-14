import pytest
import numsim_cas as cas


class TestTensorConstruction:
    def test_tensor_variable(self):
        T = cas.tensor_variable("T", dim=3, rank=2)
        assert str(T) == "T"
        assert T.dim == 3
        assert T.rank == 2

    def test_tensor_variables(self):
        result = cas.tensor_variables(("T", 3, 2), ("S", 3, 2))
        assert len(result) == 2
        assert str(result[0]) == "T"
        assert str(result[1]) == "S"

    def test_tensor_zero(self):
        z = cas.tensor_zero(dim=3, rank=2)
        assert "0" in str(z) or "zero" in str(z).lower()

    def test_kronecker_delta(self):
        d = cas.kronecker_delta(dim=3)
        assert d.dim == 3
        assert d.rank == 2

    def test_identity_tensor(self):
        I = cas.identity_tensor(dim=3, rank=2)
        assert I.dim == 3
        assert I.rank == 2


class TestTensorOperators:
    def test_add(self, T, S):
        result = T + S
        s = str(result)
        assert "T" in s and "S" in s

    def test_sub(self, T, S):
        result = T - S
        s = str(result)
        assert "T" in s

    def test_neg(self, T):
        result = -T
        assert "T" in str(result)

    def test_mul_tensor_tensor(self, T, S):
        result = T * S
        s = str(result)
        assert "T" in s and "S" in s


class TestTensorFunctions:
    def test_dev(self, T):
        result = cas.dev(T)
        assert "dev" in str(result)

    def test_inv(self, T):
        result = cas.inv(T)
        assert "inv" in str(result)

    def test_trans(self, T):
        result = cas.trans(T)
        s = str(result)
        assert "T" in s

    def test_otimes(self, T, S):
        result = cas.otimes(T, S)
        s = str(result)
        assert "T" in s and "S" in s
        assert result.rank == 4

    def test_otimesu(self, T, S):
        result = cas.otimesu(T, S)
        assert result.rank == 4

    def test_otimesl(self, T, S):
        result = cas.otimesl(T, S)
        assert result.rank == 4

    def test_inner_product(self, T, S):
        result = cas.inner_product(T, [1, 2], S, [1, 2])
        s = str(result)
        assert "T" in s and "S" in s


class TestTensorWithScalar:
    def test_scalar_mul_tensor(self, T):
        x = cas.variable("x")
        result = x * T
        assert isinstance(result, cas.TensorExpr)

    def test_tensor_mul_scalar(self, T):
        x = cas.variable("x")
        result = T * x
        assert isinstance(result, cas.TensorExpr)

    def test_tensor_div_scalar(self, T):
        x = cas.variable("x")
        result = T / x
        assert isinstance(result, cas.TensorExpr)

    def test_tensor_mul_int(self, T):
        result = T * 2
        assert isinstance(result, cas.TensorExpr)

    def test_int_mul_tensor(self, T):
        result = 2 * T
        assert isinstance(result, cas.TensorExpr)

    def test_tensor_div_int(self, T):
        result = T / 2
        assert isinstance(result, cas.TensorExpr)


class TestSequence:
    def test_create(self):
        s = cas.Sequence([1, 2, 3])
        assert len(s) == 3

    def test_getitem(self):
        s = cas.Sequence([1, 2, 3])
        assert s[0] == 1
        assert s[1] == 2
        assert s[2] == 3

    def test_repr(self):
        s = cas.Sequence([1, 2, 3])
        assert "{" in repr(s)


class TestProjectors:
    def test_P_sym(self):
        p = cas.P_sym(3)
        assert isinstance(p, cas.TensorExpr)
        assert p.rank == 4

    def test_P_skew(self):
        p = cas.P_skew(3)
        assert isinstance(p, cas.TensorExpr)
        assert p.rank == 4

    def test_P_vol(self):
        p = cas.P_vol(3)
        assert isinstance(p, cas.TensorExpr)

    def test_P_devi(self):
        p = cas.P_devi(3)
        assert isinstance(p, cas.TensorExpr)

    def test_P_harm(self):
        p = cas.P_harm(3)
        assert isinstance(p, cas.TensorExpr)


class TestTensorComparison:
    def test_equality(self):
        T1 = cas.tensor_variable("T", dim=3, rank=2)
        T2 = cas.tensor_variable("T", dim=3, rank=2)
        assert T1 == T2

    def test_inequality(self, T, S):
        assert T != S

    def test_hash(self, T):
        h = hash(T)
        assert isinstance(h, int)
