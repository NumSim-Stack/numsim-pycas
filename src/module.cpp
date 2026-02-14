#include "common.h"

PYBIND11_MODULE(_core, m) {
    m.doc() = "Python bindings for numsim-cas symbolic algebra library";

    auto scalar_cls = py::class_<ScalarExpr>(m, "ScalarExpr");
    auto tensor_cls = py::class_<TensorExpr>(m, "TensorExpr");
    auto t2s_cls    = py::class_<T2SExpr>(m, "T2SExpr");

    bind_scalar(m, scalar_cls);
    bind_tensor(m, tensor_cls);
    bind_tensor_to_scalar(m, t2s_cls);
    bind_cross_domain(m, scalar_cls, tensor_cls, t2s_cls);
}
