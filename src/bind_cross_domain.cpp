#include "common.h"

#include <numsim_cas/basic_functions.h>
#include <numsim_cas/scalar/scalar_constant.h>
#include <numsim_cas/scalar/scalar_operators.h>
#include <numsim_cas/scalar/scalar_std.h>
#include <numsim_cas/tensor/tensor_operators.h>
#include <numsim_cas/tensor/tensor_std.h>
#include <numsim_cas/tensor_to_scalar/tensor_to_scalar_operators.h>
#include <numsim_cas/tensor_to_scalar/tensor_to_scalar_std.h>
#include <numsim_cas/tensor_to_scalar/tensor_to_scalar_scalar_wrapper.h>
#include <numsim_cas/tensor_to_scalar/visitors/tensor_to_scalar_differentiation.h>

namespace cas = numsim::cas;

void bind_cross_domain(py::module_ &m,
                       py::class_<ScalarExpr> &scalar_cls,
                       py::class_<TensorExpr> &tensor_cls,
                       py::class_<T2SExpr> &t2s_cls) {
    // ScalarExpr * TensorExpr -> TensorExpr
    scalar_cls.def("__mul__", [](ScalarExpr const &a, TensorExpr const &b) -> TensorExpr {
        return a * b;
    });
    // TensorExpr * ScalarExpr -> TensorExpr
    tensor_cls.def("__mul__", [](TensorExpr const &a, ScalarExpr const &b) -> TensorExpr {
        return a * b;
    });
    // TensorExpr / ScalarExpr -> TensorExpr
    tensor_cls.def("__truediv__", [](TensorExpr const &a, ScalarExpr const &b) -> TensorExpr {
        return a / b;
    });
    // Reflected: TensorExpr.__rmul__(ScalarExpr)
    tensor_cls.def("__rmul__", [](TensorExpr const &a, ScalarExpr const &b) -> TensorExpr {
        return b * a;
    });

    // ScalarExpr + T2SExpr -> T2SExpr
    scalar_cls.def("__add__", [](ScalarExpr const &a, T2SExpr const &b) -> T2SExpr {
        return a + b;
    });
    // T2SExpr + ScalarExpr -> T2SExpr
    t2s_cls.def("__add__", [](T2SExpr const &a, ScalarExpr const &b) -> T2SExpr {
        return a + b;
    });
    // ScalarExpr - T2SExpr -> T2SExpr
    scalar_cls.def("__sub__", [](ScalarExpr const &a, T2SExpr const &b) -> T2SExpr {
        return a - b;
    });
    // T2SExpr - ScalarExpr -> T2SExpr
    t2s_cls.def("__sub__", [](T2SExpr const &a, ScalarExpr const &b) -> T2SExpr {
        return a - b;
    });

    // ScalarExpr * T2SExpr -> T2SExpr
    scalar_cls.def("__mul__", [](ScalarExpr const &a, T2SExpr const &b) -> T2SExpr {
        return a * b;
    });
    // T2SExpr * ScalarExpr -> T2SExpr
    t2s_cls.def("__mul__", [](T2SExpr const &a, ScalarExpr const &b) -> T2SExpr {
        return a * b;
    });
    // T2SExpr / ScalarExpr -> T2SExpr
    t2s_cls.def("__truediv__", [](T2SExpr const &a, ScalarExpr const &b) -> T2SExpr {
        return a / b;
    });
    // ScalarExpr / T2SExpr -> T2SExpr
    scalar_cls.def("__truediv__", [](ScalarExpr const &a, T2SExpr const &b) -> T2SExpr {
        return a / b;
    });
    // Reflected: T2SExpr.__rmul__(ScalarExpr)
    t2s_cls.def("__rmul__", [](T2SExpr const &a, ScalarExpr const &b) -> T2SExpr {
        return b * a;
    });
    // Reflected: T2SExpr.__radd__(ScalarExpr)
    t2s_cls.def("__radd__", [](T2SExpr const &a, ScalarExpr const &b) -> T2SExpr {
        return b + a;
    });

    // Differentiation: T2SExpr w.r.t. TensorExpr -> TensorExpr
    m.def("diff", [](T2SExpr const &expr, TensorExpr const &wrt) -> TensorExpr {
        cas::tensor_to_scalar_differentiation d(wrt);
        return d.apply(expr);
    }, py::arg("expr"), py::arg("wrt"),
    "Differentiate a tensor-to-scalar expression with respect to a tensor variable");

    // TensorExpr with int/float (via ScalarExpr)
    tensor_cls.def("__mul__", [](TensorExpr const &a, double b) -> TensorExpr {
        return a * cas::make_expression<cas::scalar_constant>(b);
    });
    tensor_cls.def("__mul__", [](TensorExpr const &a, int b) -> TensorExpr {
        return a * cas::make_expression<cas::scalar_constant>(b);
    });
    tensor_cls.def("__rmul__", [](TensorExpr const &a, double b) -> TensorExpr {
        return cas::make_expression<cas::scalar_constant>(b) * a;
    });
    tensor_cls.def("__rmul__", [](TensorExpr const &a, int b) -> TensorExpr {
        return cas::make_expression<cas::scalar_constant>(b) * a;
    });
    tensor_cls.def("__truediv__", [](TensorExpr const &a, double b) -> TensorExpr {
        return a / cas::make_expression<cas::scalar_constant>(b);
    });
    tensor_cls.def("__truediv__", [](TensorExpr const &a, int b) -> TensorExpr {
        return a / cas::make_expression<cas::scalar_constant>(b);
    });
}
