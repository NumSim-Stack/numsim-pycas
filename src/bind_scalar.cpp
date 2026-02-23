#include "common.h"

#include <numsim_cas/basic_functions.h>
#include <numsim_cas/scalar/scalar.h>
#include <numsim_cas/scalar/scalar_constant.h>
#include <numsim_cas/scalar/scalar_zero.h>
#include <numsim_cas/scalar/scalar_one.h>
#include <numsim_cas/scalar/scalar_globals.h>
#include <numsim_cas/scalar/scalar_functions.h>
#include <numsim_cas/scalar/scalar_operators.h>
#include <numsim_cas/scalar/scalar_std.h>
#include <numsim_cas/scalar/scalar_diff.h>
#include <numsim_cas/scalar/visitors/scalar_differentiation.h>
#include <numsim_cas/scalar/visitors/scalar_evaluator.h>
#include <numsim_cas/scalar/scalar_assume.h>

namespace cas = numsim::cas;

void bind_scalar(py::module_ &m, py::class_<ScalarExpr> &cls) {
    // Construction
    m.def("variable", [](std::string const &name) -> ScalarExpr {
        return cas::make_expression<cas::scalar>(name);
    }, py::arg("name"), "Create a scalar variable");

    m.def("variables", [](py::args names) {
        py::list result;
        for (auto &name : names) {
            result.append(cas::make_expression<cas::scalar>(name.cast<std::string>()));
        }
        return result;
    }, "Create multiple scalar variables");

    m.def("constant", [](double value) -> ScalarExpr {
        return cas::make_expression<cas::scalar_constant>(value);
    }, py::arg("value"), "Create a scalar constant");

    m.def("constant", [](int value) -> ScalarExpr {
        return cas::make_expression<cas::scalar_constant>(value);
    }, py::arg("value"), "Create a scalar constant from integer");

    // Globals
    m.attr("ZERO") = cas::get_scalar_zero();
    m.attr("ONE") = cas::get_scalar_one();

    // Helpers
    m.def("is_zero", [](ScalarExpr const &e) {
        return cas::is_same<cas::scalar_zero>(e);
    }, py::arg("expr"), "Check if expression is zero");

    m.def("is_one", [](ScalarExpr const &e) {
        return cas::is_same<cas::scalar_one>(e);
    }, py::arg("expr"), "Check if expression is one");

    m.def("is_constant", [](ScalarExpr const &e) {
        return cas::is_constant(e);
    }, py::arg("expr"), "Check if expression is a constant");

    // Assumptions — set
    m.def("assume_positive", [](ScalarExpr const &e) {
        cas::assume(e, cas::positive{});
    }, py::arg("expr"), "Assume expression is positive");
    m.def("assume_negative", [](ScalarExpr const &e) {
        cas::assume(e, cas::negative{});
    }, py::arg("expr"), "Assume expression is negative");
    m.def("assume_nonnegative", [](ScalarExpr const &e) {
        cas::assume(e, cas::nonnegative{});
    }, py::arg("expr"), "Assume expression is nonnegative");
    m.def("assume_nonpositive", [](ScalarExpr const &e) {
        cas::assume(e, cas::nonpositive{});
    }, py::arg("expr"), "Assume expression is nonpositive");
    m.def("assume_nonzero", [](ScalarExpr const &e) {
        cas::assume(e, cas::nonzero{});
    }, py::arg("expr"), "Assume expression is nonzero");
    m.def("assume_integer", [](ScalarExpr const &e) {
        cas::assume(e, cas::integer{});
    }, py::arg("expr"), "Assume expression is an integer");
    m.def("assume_real", [](ScalarExpr const &e) {
        cas::assume(e, cas::real_tag{});
    }, py::arg("expr"), "Assume expression is real");

    // Assumptions — query
    m.def("is_positive", [](ScalarExpr const &e) {
        return cas::is_positive(e);
    }, py::arg("expr"), "Check if expression is positive");
    m.def("is_negative", [](ScalarExpr const &e) {
        return cas::is_negative(e);
    }, py::arg("expr"), "Check if expression is negative");
    m.def("is_nonnegative", [](ScalarExpr const &e) {
        return cas::is_nonnegative(e);
    }, py::arg("expr"), "Check if expression is nonnegative");
    m.def("is_nonpositive", [](ScalarExpr const &e) {
        return cas::is_nonpositive(e);
    }, py::arg("expr"), "Check if expression is nonpositive");
    m.def("is_nonzero", [](ScalarExpr const &e) {
        return cas::is_nonzero(e);
    }, py::arg("expr"), "Check if expression is nonzero");

    // Differentiation
    m.def("diff", [](ScalarExpr const &expr, ScalarExpr const &wrt) -> ScalarExpr {
        return cas::diff(expr, wrt);
    }, py::arg("expr"), py::arg("wrt"), "Differentiate expr with respect to wrt");

    // Evaluation
    m.def("evaluate", [](ScalarExpr const &expr, py::dict const &values) {
        cas::scalar_evaluator<double> evaluator;
        for (auto &[key, val] : values) {
            evaluator.set(key.cast<ScalarExpr>(), val.cast<double>());
        }
        return evaluator.apply(expr);
    }, py::arg("expr"), py::arg("values"), "Evaluate expression with given variable values");

    // Scalar functions (module-level)
    m.def("sin", [](ScalarExpr const &e) -> ScalarExpr { return cas::sin(e); }, py::arg("expr"));
    m.def("cos", [](ScalarExpr const &e) -> ScalarExpr { return cas::cos(e); }, py::arg("expr"));
    m.def("tan", [](ScalarExpr const &e) -> ScalarExpr { return cas::tan(e); }, py::arg("expr"));
    m.def("asin", [](ScalarExpr const &e) -> ScalarExpr { return cas::asin(e); }, py::arg("expr"));
    m.def("acos", [](ScalarExpr const &e) -> ScalarExpr { return cas::acos(e); }, py::arg("expr"));
    m.def("atan", [](ScalarExpr const &e) -> ScalarExpr { return cas::atan(e); }, py::arg("expr"));
    m.def("exp", [](ScalarExpr const &e) -> ScalarExpr { return cas::exp(e); }, py::arg("expr"));
    m.def("log", [](ScalarExpr const &e) -> ScalarExpr { return cas::log(e); }, py::arg("expr"));
    m.def("sqrt", [](ScalarExpr const &e) -> ScalarExpr { return cas::sqrt(e); }, py::arg("expr"));
    m.def("abs", [](ScalarExpr const &e) -> ScalarExpr { return cas::abs(e); }, py::arg("expr"));
    m.def("sign", [](ScalarExpr const &e) -> ScalarExpr { return cas::sign(e); }, py::arg("expr"));
    m.def("pow", [](ScalarExpr const &base, ScalarExpr const &exp) -> ScalarExpr {
        return cas::pow(base, exp);
    }, py::arg("base"), py::arg("exp"));
    m.def("pow", [](ScalarExpr const &base, int exp) -> ScalarExpr {
        return cas::pow(base, std::move(exp));
    }, py::arg("base"), py::arg("exp"));
    m.def("pow", [](ScalarExpr const &base, double exp) -> ScalarExpr {
        return cas::pow(base, std::move(exp));
    }, py::arg("base"), py::arg("exp"));

    // String representation
    cls.def("__repr__", [](ScalarExpr const &e) {
        return cas::to_string(e);
    });
    cls.def("__str__", [](ScalarExpr const &e) {
        return cas::to_string(e);
    });

    // Comparison and hashing
    cls.def("__eq__", [](ScalarExpr const &a, ScalarExpr const &b) { return a == b; });
    cls.def("__ne__", [](ScalarExpr const &a, ScalarExpr const &b) { return a != b; });
    cls.def("__lt__", [](ScalarExpr const &a, ScalarExpr const &b) { return a < b; });
    cls.def("__gt__", [](ScalarExpr const &a, ScalarExpr const &b) { return a > b; });
    cls.def("__hash__", [](ScalarExpr const &e) {
        return static_cast<std::size_t>(e.get().hash_value());
    });

    // Arithmetic operators: ScalarExpr op ScalarExpr
    cls.def("__add__", [](ScalarExpr const &a, ScalarExpr const &b) -> ScalarExpr { return a + b; });
    cls.def("__sub__", [](ScalarExpr const &a, ScalarExpr const &b) -> ScalarExpr { return a - b; });
    cls.def("__mul__", [](ScalarExpr const &a, ScalarExpr const &b) -> ScalarExpr { return a * b; });
    cls.def("__truediv__", [](ScalarExpr const &a, ScalarExpr const &b) -> ScalarExpr { return a / b; });
    cls.def("__neg__", [](ScalarExpr const &a) -> ScalarExpr { return -a; });
    cls.def("__pow__", [](ScalarExpr const &a, ScalarExpr const &b) -> ScalarExpr { return cas::pow(a, b); });
    cls.def("__pow__", [](ScalarExpr const &a, int b) -> ScalarExpr { return cas::pow(a, std::move(b)); });
    cls.def("__pow__", [](ScalarExpr const &a, double b) -> ScalarExpr { return cas::pow(a, std::move(b)); });

    // In-place operators
    cls.def("__iadd__", [](ScalarExpr &a, ScalarExpr const &b) -> ScalarExpr & { a += b; return a; });
    cls.def("__imul__", [](ScalarExpr &a, ScalarExpr const &b) -> ScalarExpr & { a *= b; return a; });

    // Arithmetic with Python numeric types (right-hand side)
    cls.def("__add__", [](ScalarExpr const &a, double b) -> ScalarExpr {
        return a + cas::make_expression<cas::scalar_constant>(b);
    });
    cls.def("__sub__", [](ScalarExpr const &a, double b) -> ScalarExpr {
        return a - cas::make_expression<cas::scalar_constant>(b);
    });
    cls.def("__mul__", [](ScalarExpr const &a, double b) -> ScalarExpr {
        return a * cas::make_expression<cas::scalar_constant>(b);
    });
    cls.def("__truediv__", [](ScalarExpr const &a, double b) -> ScalarExpr {
        return a / cas::make_expression<cas::scalar_constant>(b);
    });
    cls.def("__add__", [](ScalarExpr const &a, int b) -> ScalarExpr {
        return a + cas::make_expression<cas::scalar_constant>(b);
    });
    cls.def("__sub__", [](ScalarExpr const &a, int b) -> ScalarExpr {
        return a - cas::make_expression<cas::scalar_constant>(b);
    });
    cls.def("__mul__", [](ScalarExpr const &a, int b) -> ScalarExpr {
        return a * cas::make_expression<cas::scalar_constant>(b);
    });
    cls.def("__truediv__", [](ScalarExpr const &a, int b) -> ScalarExpr {
        return a / cas::make_expression<cas::scalar_constant>(b);
    });

    // Reflected operators (numeric op ScalarExpr)
    cls.def("__radd__", [](ScalarExpr const &a, double b) -> ScalarExpr {
        return cas::make_expression<cas::scalar_constant>(b) + a;
    });
    cls.def("__rsub__", [](ScalarExpr const &a, double b) -> ScalarExpr {
        return cas::make_expression<cas::scalar_constant>(b) - a;
    });
    cls.def("__rmul__", [](ScalarExpr const &a, double b) -> ScalarExpr {
        return cas::make_expression<cas::scalar_constant>(b) * a;
    });
    cls.def("__rtruediv__", [](ScalarExpr const &a, double b) -> ScalarExpr {
        return cas::make_expression<cas::scalar_constant>(b) / a;
    });
    cls.def("__radd__", [](ScalarExpr const &a, int b) -> ScalarExpr {
        return cas::make_expression<cas::scalar_constant>(b) + a;
    });
    cls.def("__rsub__", [](ScalarExpr const &a, int b) -> ScalarExpr {
        return cas::make_expression<cas::scalar_constant>(b) - a;
    });
    cls.def("__rmul__", [](ScalarExpr const &a, int b) -> ScalarExpr {
        return cas::make_expression<cas::scalar_constant>(b) * a;
    });
    cls.def("__rtruediv__", [](ScalarExpr const &a, int b) -> ScalarExpr {
        return cas::make_expression<cas::scalar_constant>(b) / a;
    });
}
