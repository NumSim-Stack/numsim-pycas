#include "common.h"

#include <numsim_cas/basic_functions.h>
#include <numsim_cas/tensor/tensor.h>
#include <numsim_cas/tensor/tensor_zero.h>
#include <numsim_cas/tensor/kronecker_delta.h>
#include <numsim_cas/tensor/identity_tensor.h>
#include <numsim_cas/tensor/projection_tensor.h>
#include <numsim_cas/tensor/tensor_operators.h>
#include <numsim_cas/tensor/tensor_functions.h>
#include <numsim_cas/tensor/tensor_std.h>
#include <numsim_cas/tensor/tensor_negative.h>
#include <numsim_cas/tensor/functions/tensor_pow.h>
#include <numsim_cas/scalar/scalar_constant.h>
#include <numsim_cas/tensor/sequence.h>

namespace cas = numsim::cas;

namespace {

cas::sequence list_to_sequence(py::list const &lst) {
    std::vector<cas::sequence::index_t> indices;
    indices.reserve(lst.size());
    for (auto &item : lst) {
        auto val = item.cast<cas::sequence::index_t>();
        if (val == 0)
            throw std::out_of_range("sequence: 1-based index cannot be 0");
        indices.push_back(val - 1);
    }
    cas::sequence s(indices.size());
    for (std::size_t i = 0; i < indices.size(); ++i) {
        s[i] = indices[i];
    }
    return s;
}

} // namespace

void bind_tensor(py::module_ &m, py::class_<TensorExpr> &cls) {
    // Sequence class
    py::class_<cas::sequence>(m, "Sequence")
        .def(py::init([](py::list indices) {
            return list_to_sequence(indices);
        }), py::arg("indices"), "Create a sequence from a list of 1-based indices")
        .def("__len__", &cas::sequence::size)
        .def("__getitem__", [](cas::sequence const &s, std::size_t i) {
            if (i >= s.size())
                throw py::index_error("index out of range");
            return s[i]; // returns 1-based
        })
        .def("__repr__", [](cas::sequence const &s) {
            std::ostringstream os;
            os << s;
            return os.str();
        });

    // Tensor construction
    m.def("tensor_variable", [](std::string const &name, std::size_t dim, std::size_t rank) -> TensorExpr {
        return cas::make_expression<cas::tensor>(name, dim, rank);
    }, py::arg("name"), py::arg("dim"), py::arg("rank"),
    "Create a tensor variable");

    m.def("tensor_variables", [](py::args args) {
        py::list result;
        for (auto &arg : args) {
            auto tup = arg.cast<py::tuple>();
            result.append(cas::make_expression<cas::tensor>(
                tup[0].cast<std::string>(),
                tup[1].cast<std::size_t>(),
                tup[2].cast<std::size_t>()));
        }
        return result;
    }, "Create multiple tensor variables from (name, dim, rank) tuples");

    // Special tensors
    m.def("tensor_zero", [](std::size_t dim, std::size_t rank) -> TensorExpr {
        return cas::make_expression<cas::tensor_zero>(dim, rank);
    }, py::arg("dim"), py::arg("rank"), "Create a zero tensor");

    m.def("kronecker_delta", [](std::size_t dim) -> TensorExpr {
        return cas::make_expression<cas::kronecker_delta>(dim);
    }, py::arg("dim"), "Create a Kronecker delta tensor");

    m.def("identity_tensor", [](std::size_t dim, std::size_t rank) -> TensorExpr {
        return cas::make_expression<cas::identity_tensor>(dim, rank);
    }, py::arg("dim"), py::arg("rank"), "Create an identity tensor");

    // Properties
    cls.def_property_readonly("dim", [](TensorExpr const &e) { return e.get().dim(); });
    cls.def_property_readonly("rank", [](TensorExpr const &e) { return e.get().rank(); });

    // String representation
    cls.def("__repr__", [](TensorExpr &e) { return cas::to_string(e); });
    cls.def("__str__", [](TensorExpr &e) { return cas::to_string(e); });

    // Comparison and hashing
    cls.def("__eq__", [](TensorExpr const &a, TensorExpr const &b) { return a == b; });
    cls.def("__ne__", [](TensorExpr const &a, TensorExpr const &b) { return a != b; });
    cls.def("__hash__", [](TensorExpr const &e) {
        return static_cast<std::size_t>(e.get().hash_value());
    });

    // Tensor-tensor operators
    cls.def("__add__", [](TensorExpr const &a, TensorExpr const &b) -> TensorExpr { return a + b; });
    cls.def("__sub__", [](TensorExpr const &a, TensorExpr const &b) -> TensorExpr { return a - b; });
    cls.def("__mul__", [](TensorExpr const &a, TensorExpr const &b) -> TensorExpr { return a * b; });
    cls.def("__neg__", [](TensorExpr const &a) -> TensorExpr { return -a; });

    // Tensor functions (module-level)
    m.def("dev", [](TensorExpr const &e) -> TensorExpr {
        return cas::dev(e);
    }, py::arg("expr"), "Deviatoric part of a rank-2 tensor");

    m.def("inv", [](TensorExpr const &e) -> TensorExpr {
        return cas::inv(e);
    }, py::arg("expr"), "Inverse of a tensor");

    m.def("trans", [](TensorExpr const &e) -> TensorExpr {
        return cas::trans(e);
    }, py::arg("expr"), "Transpose of a tensor");

    m.def("inner_product", [](TensorExpr const &lhs, py::list lhs_idx,
                              TensorExpr const &rhs, py::list rhs_idx) -> TensorExpr {
        return cas::inner_product(lhs, list_to_sequence(lhs_idx),
                                  rhs, list_to_sequence(rhs_idx));
    }, py::arg("lhs"), py::arg("lhs_indices"),
       py::arg("rhs"), py::arg("rhs_indices"),
    "Inner product with explicit index contraction");

    m.def("otimes", [](TensorExpr const &lhs, TensorExpr const &rhs) -> TensorExpr {
        return cas::otimes(lhs, rhs);
    }, py::arg("lhs"), py::arg("rhs"), "Outer product (tensor product)");

    m.def("otimesu", [](TensorExpr const &lhs, TensorExpr const &rhs) -> TensorExpr {
        return cas::otimesu(lhs, rhs);
    }, py::arg("lhs"), py::arg("rhs"), "Upper outer product");

    m.def("otimesl", [](TensorExpr const &lhs, TensorExpr const &rhs) -> TensorExpr {
        return cas::otimesl(lhs, rhs);
    }, py::arg("lhs"), py::arg("rhs"), "Lower outer product");

    // Note: permute_indices omitted due to upstream compile issues in tensor_functions.h

    // Projector presets
    m.def("P_sym", [](std::size_t d) -> TensorExpr { return cas::P_sym(d); },
          py::arg("dim"), "Symmetric projector");
    m.def("P_skew", [](std::size_t d) -> TensorExpr { return cas::P_skew(d); },
          py::arg("dim"), "Skew-symmetric projector");
    m.def("P_vol", [](std::size_t d) -> TensorExpr { return cas::P_vol(d); },
          py::arg("dim"), "Volumetric projector");
    m.def("P_devi", [](std::size_t d) -> TensorExpr { return cas::P_devi(d); },
          py::arg("dim"), "Deviatoric projector");
    m.def("P_harm", [](std::size_t d, std::size_t r) -> TensorExpr { return cas::P_harm(d, r); },
          py::arg("dim"), py::arg("rank") = 2, "Harmonic projector");

    // Tensor pow
    m.def("pow", [](TensorExpr const &base, ScalarExpr const &exp) -> TensorExpr {
        return cas::make_expression<cas::tensor_pow>(base, exp);
    }, py::arg("base"), py::arg("exp"));
    m.def("pow", [](TensorExpr const &base, int exp) -> TensorExpr {
        return cas::make_expression<cas::tensor_pow>(
            base, cas::make_expression<cas::scalar_constant>(exp));
    }, py::arg("base"), py::arg("exp"));
}
