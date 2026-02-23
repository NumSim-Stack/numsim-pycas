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

    def test_pow_int(self, T):
        result = T ** 2
        assert isinstance(result, cas.TensorExpr)
        assert "T" in str(result)

    def test_pow_scalar(self, T):
        n = cas.variable("n")
        result = T ** n
        assert isinstance(result, cas.TensorExpr)
        assert "T" in str(result) and "n" in str(result)


class TestTensorFunctions:
    def test_dev(self, T):
        result = cas.dev(T)
        assert "dev" in str(result)

    def test_sym(self, T):
        result = cas.sym(T)
        assert isinstance(result, cas.TensorExpr)
        assert result.rank == 2
        assert "T" in str(result)

    def test_vol(self, T):
        result = cas.vol(T)
        assert isinstance(result, cas.TensorExpr)
        assert result.rank == 2
        assert "T" in str(result)

    def test_skew(self, T):
        result = cas.skew(T)
        assert isinstance(result, cas.TensorExpr)
        assert result.rank == 2
        assert "T" in str(result)

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


class TestTensorDifferentiation:
    def test_diff_self(self, T):
        """dT/dT should be rank-4 identity."""
        result = cas.diff(T, T)
        assert isinstance(result, cas.TensorExpr)
        assert result.rank == 4

    def test_diff_other(self, T, S):
        """dT/dS should be zero (independent variables)."""
        result = cas.diff(T, S)
        assert isinstance(result, cas.TensorExpr)

    def test_diff_trace(self):
        """d(tr(C))/dC = I (Kronecker delta)."""
        C = cas.tensor_variable("C", dim=3, rank=2)
        trC = cas.trace(C)
        result = cas.diff(trC, C)
        assert isinstance(result, cas.TensorExpr)
        assert result.rank == 2
        assert str(result) == "I"

    def test_diff_dot(self):
        """d(C:C)/dC = 2*C."""
        C = cas.tensor_variable("C", dim=3, rank=2)
        dotC = cas.dot(C)
        result = cas.diff(dotC, C)
        assert "2" in str(result) and "C" in str(result)

    def test_diff_trace_squared(self):
        """d(tr(C)^2)/dC = 2*tr(C)*I."""
        C = cas.tensor_variable("C", dim=3, rank=2)
        trC = cas.trace(C)
        result = cas.diff(trC ** 2, C)
        s = str(result)
        assert "tr" in s and "I" in s


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


class TestTensorSpaceAssumptions:
    def test_assume_symmetric_query(self):
        C = cas.tensor_variable("C", dim=3, rank=2)
        assert not cas.is_symmetric(C)
        cas.assume_symmetric(C)
        assert cas.is_symmetric(C)

    def test_assume_skew_query(self):
        W = cas.tensor_variable("W", dim=3, rank=2)
        cas.assume_skew(W)
        assert cas.is_skew(W)
        assert not cas.is_symmetric(W)

    def test_sym_of_symmetric_is_identity(self):
        C = cas.tensor_variable("C", dim=3, rank=2)
        cas.assume_symmetric(C)
        result = cas.sym(C)
        assert result == C

    def test_skew_of_symmetric_is_zero(self):
        C = cas.tensor_variable("C", dim=3, rank=2)
        cas.assume_symmetric(C)
        result = cas.skew(C)
        z = cas.tensor_zero(dim=3, rank=2)
        assert result == z

    def test_sym_of_skew_is_zero(self):
        W = cas.tensor_variable("W", dim=3, rank=2)
        cas.assume_skew(W)
        result = cas.sym(W)
        z = cas.tensor_zero(dim=3, rank=2)
        assert result == z

    def test_skew_of_skew_is_identity(self):
        W = cas.tensor_variable("W", dim=3, rank=2)
        cas.assume_skew(W)
        result = cas.skew(W)
        assert result == W

    def test_dev_vol_unaffected_by_symmetric(self):
        """dev/vol of a merely-symmetric tensor still applies the projection."""
        C = cas.tensor_variable("C", dim=3, rank=2)
        cas.assume_symmetric(C)
        dev_C = cas.dev(C)
        vol_C = cas.vol(C)
        # dev and vol should NOT be identity on a merely-symmetric tensor
        assert dev_C != C
        assert vol_C != C

    def test_assume_volumetric(self):
        V = cas.tensor_variable("V", dim=3, rank=2)
        cas.assume_volumetric(V)
        assert cas.is_volumetric(V)
        assert cas.is_symmetric(V)  # volumetric implies symmetric
        assert not cas.is_deviatoric(V)

    def test_vol_of_volumetric_is_identity(self):
        V = cas.tensor_variable("V", dim=3, rank=2)
        cas.assume_volumetric(V)
        assert cas.vol(V) == V

    def test_dev_of_volumetric_is_zero(self):
        V = cas.tensor_variable("V", dim=3, rank=2)
        cas.assume_volumetric(V)
        result = cas.dev(V)
        z = cas.tensor_zero(dim=3, rank=2)
        assert result == z

    def test_assume_deviatoric(self):
        D = cas.tensor_variable("D", dim=3, rank=2)
        cas.assume_deviatoric(D)
        assert cas.is_deviatoric(D)
        assert cas.is_symmetric(D)  # deviatoric implies symmetric
        assert not cas.is_volumetric(D)

    def test_dev_of_deviatoric_is_identity(self):
        D = cas.tensor_variable("D", dim=3, rank=2)
        cas.assume_deviatoric(D)
        assert cas.dev(D) == D

    def test_vol_of_deviatoric_is_zero(self):
        D = cas.tensor_variable("D", dim=3, rank=2)
        cas.assume_deviatoric(D)
        result = cas.vol(D)
        z = cas.tensor_zero(dim=3, rank=2)
        assert result == z

    def test_sym_of_volumetric_is_identity(self):
        """sym(volumetric) should return the tensor (vol is subspace of sym)."""
        V = cas.tensor_variable("V", dim=3, rank=2)
        cas.assume_volumetric(V)
        assert cas.sym(V) == V

    def test_sym_of_deviatoric_is_identity(self):
        """sym(deviatoric) should return the tensor (dev is subspace of sym)."""
        D = cas.tensor_variable("D", dim=3, rank=2)
        cas.assume_deviatoric(D)
        assert cas.sym(D) == D

    def test_skew_of_volumetric_is_zero(self):
        V = cas.tensor_variable("V", dim=3, rank=2)
        cas.assume_volumetric(V)
        result = cas.skew(V)
        z = cas.tensor_zero(dim=3, rank=2)
        assert result == z

    def test_skew_of_deviatoric_is_zero(self):
        D = cas.tensor_variable("D", dim=3, rank=2)
        cas.assume_deviatoric(D)
        result = cas.skew(D)
        z = cas.tensor_zero(dim=3, rank=2)
        assert result == z

    def test_no_assumption_by_default(self):
        T = cas.tensor_variable("T", dim=3, rank=2)
        assert not cas.is_symmetric(T)
        assert not cas.is_skew(T)
        assert not cas.is_volumetric(T)
        assert not cas.is_deviatoric(T)


class TestSpacePropagation:
    """Tests for tensor_space propagation through derived expressions."""

    def test_sym_of_2C(self):
        """sym(2*C) → 2*C when C is symmetric."""
        C = cas.tensor_variable("C", dim=3, rank=2)
        cas.assume_symmetric(C)
        result = cas.sym(2 * C)
        assert result == 2 * C

    def test_skew_of_neg_C_is_zero(self):
        """skew(-C) → 0 when C is symmetric."""
        C = cas.tensor_variable("C", dim=3, rank=2)
        cas.assume_symmetric(C)
        result = cas.skew(-C)
        z = cas.tensor_zero(dim=3, rank=2)
        assert result == z

    def test_sym_of_C_plus_C(self):
        """sym(C+C) → 2*C when C is symmetric (via add propagation)."""
        C = cas.tensor_variable("C", dim=3, rank=2)
        cas.assume_symmetric(C)
        result = cas.sym(C + C)
        assert result == 2 * C

    def test_dev_of_2C(self):
        """dev(2*C) → 2*C when C is deviatoric."""
        D = cas.tensor_variable("D", dim=3, rank=2)
        cas.assume_deviatoric(D)
        result = cas.dev(2 * D)
        assert result == 2 * D


class TestDiffWithAssumptions:
    """Tests for differentiation with tensor space assumptions."""

    def test_diff_symmetric_self(self):
        """dC/dC → P_sym when C is symmetric."""
        C = cas.tensor_variable("C", dim=3, rank=2)
        cas.assume_symmetric(C)
        result = cas.diff(C, C)
        assert result.rank == 4
        assert str(result) == "P_sym{4}"

    def test_diff_trace_symmetric(self):
        """d(tr(C))/dC → I when C is symmetric."""
        C = cas.tensor_variable("C", dim=3, rank=2)
        cas.assume_symmetric(C)
        trC = cas.trace(C)
        result = cas.diff(trC, C)
        assert result.rank == 2
        assert str(result) == "I"

    def test_diff_dot_symmetric(self):
        """d(C:C)/dC → 2*C when C is symmetric."""
        C = cas.tensor_variable("C", dim=3, rank=2)
        cas.assume_symmetric(C)
        dotC = cas.dot(C)
        result = cas.diff(dotC, C)
        assert str(result) == "2*C"

    def test_diff_trace_squared_symmetric(self):
        """d(tr(C)^2)/dC → 2*tr(C)*I when C is symmetric."""
        C = cas.tensor_variable("C", dim=3, rank=2)
        cas.assume_symmetric(C)
        trC = cas.trace(C)
        result = cas.diff(trC ** 2, C)
        s = str(result)
        assert "tr" in s and "I" in s


class TestKroneckerDeltaProjection:
    """Tests for projection of identity tensor (kronecker delta)."""

    def test_sym_of_I(self):
        """sym(I) → I."""
        I = cas.kronecker_delta(dim=3)
        result = cas.sym(I)
        assert result == I

    def test_vol_of_I(self):
        """vol(I) → I."""
        I = cas.kronecker_delta(dim=3)
        result = cas.vol(I)
        assert result == I

    def test_dev_of_I(self):
        """dev(I) → 0."""
        I = cas.kronecker_delta(dim=3)
        result = cas.dev(I)
        z = cas.tensor_zero(dim=3, rank=2)
        assert result == z

    def test_skew_of_I(self):
        """skew(I) → 0."""
        I = cas.kronecker_delta(dim=3)
        result = cas.skew(I)
        z = cas.tensor_zero(dim=3, rank=2)
        assert result == z
