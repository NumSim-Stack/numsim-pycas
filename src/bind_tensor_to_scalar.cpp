#include "common.h"

#include <numsim_cas/basic_functions.h>
#include <numsim_cas/tensor_to_scalar/tensor_to_scalar_expression.h>
#include <numsim_cas/tensor_to_scalar/tensor_to_scalar_functions.h>
#include <numsim_cas/tensor_to_scalar/tensor_to_scalar_operators.h>
#include <numsim_cas/tensor_to_scalar/tensor_to_scalar_std.h>
#include <numsim_cas/tensor_to_scalar/tensor_to_scalar_zero.h>
#include <numsim_cas/tensor_to_scalar/tensor_to_scalar_one.h>
#include <numsim_cas/tensor_to_scalar/tensor_to_scalar_scalar_wrapper.h>
#include <numsim_cas/scalar/scalar_constant.h>
#include <numsim_cas/tensor/sequence.h>

namespace cas = numsim::cas;

void bind_tensor_to_scalar(py::module_ &m, py::class_<T2SExpr> &cls) {
    // T2S construction functions (tensor -> scalar)
    m.def("trace", [](TensorExpr const &e) -> T2SExpr {
        return cas::trace(e);
    }, py::arg("expr"), "Trace of a rank-2 tensor");

    m.def("det", [](TensorExpr const &e) -> T2SExpr {
        return cas::det(e);
    }, py::arg("expr"), "Determinant of a rank-2 tensor");

    m.def("norm", [](TensorExpr const &e) -> T2SExpr {
        return cas::norm(e);
    }, py::arg("expr"), "Frobenius norm of a rank-2 tensor");

    m.def("dot", [](TensorExpr const &e) -> T2SExpr {
        return cas::dot(e);
    }, py::arg("expr"), "Dot contraction of a tensor");

    m.def("dot_product", [](TensorExpr const &lhs, py::list lhs_idx,
                            TensorExpr const &rhs, py::list rhs_idx) -> T2SExpr {
        std::vector<cas::sequence::index_t> lhs_v, rhs_v;
        for (auto &item : lhs_idx) {
            auto val = item.cast<cas::sequence::index_t>();
            if (val == 0) throw std::out_of_range("sequence: 1-based index cannot be 0");
            lhs_v.push_back(val - 1);
        }
        for (auto &item : rhs_idx) {
            auto val = item.cast<cas::sequence::index_t>();
            if (val == 0) throw std::out_of_range("sequence: 1-based index cannot be 0");
            rhs_v.push_back(val - 1);
        }
        cas::sequence sl(lhs_v.size()), sr(rhs_v.size());
        for (std::size_t i = 0; i < lhs_v.size(); ++i) sl[i] = lhs_v[i];
        for (std::size_t i = 0; i < rhs_v.size(); ++i) sr[i] = rhs_v[i];
        return cas::dot_product(lhs, std::move(sl), rhs, std::move(sr));
    }, py::arg("lhs"), py::arg("lhs_indices"),
       py::arg("rhs"), py::arg("rhs_indices"),
    "Full contraction of two tensors to scalar");

    // String representation
    cls.def("__repr__", [](T2SExpr const &e) { return cas::to_string(e); });
    cls.def("__str__", [](T2SExpr const &e) { return cas::to_string(e); });

    // Comparison and hashing
    cls.def("__eq__", [](T2SExpr const &a, T2SExpr const &b) { return a == b; });
    cls.def("__ne__", [](T2SExpr const &a, T2SExpr const &b) { return a != b; });
    cls.def("__hash__", [](T2SExpr const &e) {
        return static_cast<std::size_t>(e.get().hash_value());
    });

    // T2S-T2S operators
    cls.def("__add__", [](T2SExpr const &a, T2SExpr const &b) -> T2SExpr { return a + b; });
    cls.def("__sub__", [](T2SExpr const &a, T2SExpr const &b) -> T2SExpr { return a - b; });
    cls.def("__mul__", [](T2SExpr const &a, T2SExpr const &b) -> T2SExpr { return a * b; });
    cls.def("__truediv__", [](T2SExpr const &a, T2SExpr const &b) -> T2SExpr { return a / b; });
    cls.def("__neg__", [](T2SExpr const &a) -> T2SExpr { return -a; });
    cls.def("__pow__", [](T2SExpr const &a, T2SExpr const &b) -> T2SExpr { return cas::pow(a, b); });
    cls.def("__pow__", [](T2SExpr const &a, int b) -> T2SExpr { return cas::pow(a, std::move(b)); });
    cls.def("__pow__", [](T2SExpr const &a, double b) -> T2SExpr { return cas::pow(a, std::move(b)); });

    // T2S with numeric
    cls.def("__add__", [](T2SExpr const &a, double b) -> T2SExpr {
        return a + cas::make_expression<cas::tensor_to_scalar_scalar_wrapper>(
            cas::make_expression<cas::scalar_constant>(b));
    });
    cls.def("__sub__", [](T2SExpr const &a, double b) -> T2SExpr {
        return a - cas::make_expression<cas::tensor_to_scalar_scalar_wrapper>(
            cas::make_expression<cas::scalar_constant>(b));
    });
    cls.def("__mul__", [](T2SExpr const &a, double b) -> T2SExpr {
        return a * cas::make_expression<cas::tensor_to_scalar_scalar_wrapper>(
            cas::make_expression<cas::scalar_constant>(b));
    });
    cls.def("__truediv__", [](T2SExpr const &a, double b) -> T2SExpr {
        return a / cas::make_expression<cas::tensor_to_scalar_scalar_wrapper>(
            cas::make_expression<cas::scalar_constant>(b));
    });
    cls.def("__add__", [](T2SExpr const &a, int b) -> T2SExpr {
        return a + cas::make_expression<cas::tensor_to_scalar_scalar_wrapper>(
            cas::make_expression<cas::scalar_constant>(b));
    });
    cls.def("__sub__", [](T2SExpr const &a, int b) -> T2SExpr {
        return a - cas::make_expression<cas::tensor_to_scalar_scalar_wrapper>(
            cas::make_expression<cas::scalar_constant>(b));
    });
    cls.def("__mul__", [](T2SExpr const &a, int b) -> T2SExpr {
        return a * cas::make_expression<cas::tensor_to_scalar_scalar_wrapper>(
            cas::make_expression<cas::scalar_constant>(b));
    });
    cls.def("__truediv__", [](T2SExpr const &a, int b) -> T2SExpr {
        return a / cas::make_expression<cas::tensor_to_scalar_scalar_wrapper>(
            cas::make_expression<cas::scalar_constant>(b));
    });

    // Reflected numeric ops
    cls.def("__radd__", [](T2SExpr const &a, double b) -> T2SExpr {
        return cas::make_expression<cas::tensor_to_scalar_scalar_wrapper>(
            cas::make_expression<cas::scalar_constant>(b)) + a;
    });
    cls.def("__rsub__", [](T2SExpr const &a, double b) -> T2SExpr {
        return cas::make_expression<cas::tensor_to_scalar_scalar_wrapper>(
            cas::make_expression<cas::scalar_constant>(b)) - a;
    });
    cls.def("__rmul__", [](T2SExpr const &a, double b) -> T2SExpr {
        return cas::make_expression<cas::tensor_to_scalar_scalar_wrapper>(
            cas::make_expression<cas::scalar_constant>(b)) * a;
    });
    cls.def("__radd__", [](T2SExpr const &a, int b) -> T2SExpr {
        return cas::make_expression<cas::tensor_to_scalar_scalar_wrapper>(
            cas::make_expression<cas::scalar_constant>(b)) + a;
    });
    cls.def("__rsub__", [](T2SExpr const &a, int b) -> T2SExpr {
        return cas::make_expression<cas::tensor_to_scalar_scalar_wrapper>(
            cas::make_expression<cas::scalar_constant>(b)) - a;
    });
    cls.def("__rmul__", [](T2SExpr const &a, int b) -> T2SExpr {
        return cas::make_expression<cas::tensor_to_scalar_scalar_wrapper>(
            cas::make_expression<cas::scalar_constant>(b)) * a;
    });

    // T2S functions
    m.def("log", [](T2SExpr const &e) -> T2SExpr { return cas::log(e); }, py::arg("expr"));
    m.def("pow", [](T2SExpr const &base, T2SExpr const &exp) -> T2SExpr { return cas::pow(base, exp); },
          py::arg("base"), py::arg("exp"));
    m.def("pow", [](T2SExpr const &base, int exp) -> T2SExpr { return cas::pow(base, std::move(exp)); },
          py::arg("base"), py::arg("exp"));
    m.def("pow", [](T2SExpr const &base, double exp) -> T2SExpr { return cas::pow(base, std::move(exp)); },
          py::arg("base"), py::arg("exp"));
    m.def("pow", [](T2SExpr const &base, ScalarExpr const &exp) -> T2SExpr { return cas::pow(base, exp); },
          py::arg("base"), py::arg("exp"));
}
