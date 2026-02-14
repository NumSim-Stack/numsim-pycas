import pytest
import numsim_cas as cas


class TestScalarConstruction:
    def test_variable(self):
        x = cas.variable("x")
        assert str(x) == "x"

    def test_variables(self):
        result = cas.variables("x", "y", "z")
        assert len(result) == 3
        assert str(result[0]) == "x"
        assert str(result[1]) == "y"
        assert str(result[2]) == "z"

    def test_constant_int(self):
        c = cas.constant(5)
        assert "5" in str(c)

    def test_constant_float(self):
        c = cas.constant(3.14)
        assert "3.14" in str(c)

    def test_globals(self):
        assert cas.is_zero(cas.ZERO)
        assert cas.is_one(cas.ONE)


class TestScalarOperators:
    def test_add(self, x, y):
        result = x + y
        s = str(result)
        assert "x" in s and "y" in s

    def test_sub(self, x, y):
        result = x - y
        s = str(result)
        assert "x" in s

    def test_mul(self, x, y):
        result = x * y
        s = str(result)
        assert "x" in s and "y" in s

    def test_div(self, x, y):
        result = x / y
        s = str(result)
        assert "x" in s and "y" in s

    def test_neg(self, x):
        result = -x
        s = str(result)
        assert "x" in s

    def test_pow(self, x):
        result = x ** 2
        s = str(result)
        assert "x" in s

    def test_add_same(self, x):
        result = x + x
        s = str(result)
        # should simplify to 2*x
        assert "2" in s and "x" in s

    def test_iadd(self, x, y):
        result = cas.variable("x")
        result += y
        s = str(result)
        assert "x" in s and "y" in s

    def test_imul(self, x, y):
        result = cas.variable("x")
        result *= y
        s = str(result)
        assert "x" in s and "y" in s


class TestScalarWithNumeric:
    def test_add_int(self, x):
        result = x + 1
        assert "x" in str(result)

    def test_radd_int(self, x):
        result = 1 + x
        assert "x" in str(result)

    def test_mul_int(self, x):
        result = x * 2
        assert "x" in str(result)

    def test_rmul_int(self, x):
        result = 2 * x
        assert "x" in str(result)

    def test_sub_int(self, x):
        result = x - 3
        assert "x" in str(result)

    def test_rsub_int(self, x):
        result = 3 - x
        assert "x" in str(result)

    def test_div_float(self, x):
        result = x / 2.0
        assert "x" in str(result)

    def test_rdiv_int(self, x):
        result = 1 / x
        assert "x" in str(result)


class TestScalarFunctions:
    def test_sin(self, x):
        result = cas.sin(x)
        assert "sin" in str(result)

    def test_cos(self, x):
        result = cas.cos(x)
        assert "cos" in str(result)

    def test_tan(self, x):
        result = cas.tan(x)
        assert "tan" in str(result)

    def test_asin(self, x):
        result = cas.asin(x)
        assert "asin" in str(result)

    def test_acos(self, x):
        result = cas.acos(x)
        assert "acos" in str(result)

    def test_atan(self, x):
        result = cas.atan(x)
        assert "atan" in str(result)

    def test_exp(self, x):
        result = cas.exp(x)
        assert "exp" in str(result)

    def test_log(self, x):
        result = cas.log(x)
        assert "log" in str(result) or "ln" in str(result)

    def test_sqrt(self, x):
        result = cas.sqrt(x)
        assert "sqrt" in str(result)

    def test_abs(self, x):
        result = cas.abs(x)
        assert "abs" in str(result)

    def test_sign(self, x):
        result = cas.sign(x)
        assert "sign" in str(result) or "sgn" in str(result)

    def test_pow_function(self, x):
        result = cas.pow(x, 3)
        assert "x" in str(result)


class TestScalarDifferentiation:
    def test_diff_variable(self, x):
        result = cas.diff(x, x)
        assert cas.is_one(result)

    def test_diff_other_variable(self, x, y):
        result = cas.diff(x, y)
        assert cas.is_zero(result)

    def test_diff_sum(self, x, y):
        expr = x + y
        result = cas.diff(expr, x)
        assert cas.is_one(result)

    def test_diff_product(self, x, y):
        expr = x * y
        result = cas.diff(expr, x)
        # d(x*y)/dx = y
        assert str(result) == "y" or "y" in str(result)

    def test_diff_power(self, x):
        expr = x ** 2
        result = cas.diff(expr, x)
        s = str(result)
        assert "x" in s and "2" in s

    def test_diff_sin(self, x):
        expr = cas.sin(x)
        result = cas.diff(expr, x)
        assert "cos" in str(result)


class TestScalarEvaluation:
    def test_evaluate_variable(self, x):
        result = cas.evaluate(x, {x: 5.0})
        assert result == pytest.approx(5.0)

    def test_evaluate_constant(self):
        c = cas.constant(3.14)
        result = cas.evaluate(c, {})
        assert result == pytest.approx(3.14)

    def test_evaluate_sum(self, x, y):
        expr = x + y
        result = cas.evaluate(expr, {x: 1.0, y: 2.0})
        assert result == pytest.approx(3.0)

    def test_evaluate_product(self, x, y):
        expr = x * y
        result = cas.evaluate(expr, {x: 3.0, y: 4.0})
        assert result == pytest.approx(12.0)

    def test_evaluate_power(self, x):
        expr = x ** 3
        result = cas.evaluate(expr, {x: 2.0})
        assert result == pytest.approx(8.0)

    def test_evaluate_complex_expr(self, x, y):
        expr = x ** 2 + 2 * x * y
        result = cas.evaluate(expr, {x: 1.0, y: 2.0})
        assert result == pytest.approx(5.0)

    def test_evaluate_sin(self, x):
        import math
        expr = cas.sin(x)
        result = cas.evaluate(expr, {x: math.pi / 2})
        assert result == pytest.approx(1.0)


class TestScalarComparison:
    def test_equality(self, x):
        x2 = cas.variable("x")
        assert x == x2

    def test_inequality(self, x, y):
        assert x != y

    def test_hash(self, x):
        h = hash(x)
        assert isinstance(h, int)

    def test_hash_consistency(self, x):
        x2 = cas.variable("x")
        assert hash(x) == hash(x2)
