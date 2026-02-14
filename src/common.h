#ifndef NUMSIM_CAS_PYTHON_COMMON_H
#define NUMSIM_CAS_PYTHON_COMMON_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <numsim_cas/numsim_cas.h>

namespace py = pybind11;

using ScalarExpr = numsim::cas::expression_holder<numsim::cas::scalar_expression>;
using TensorExpr = numsim::cas::expression_holder<numsim::cas::tensor_expression>;
using T2SExpr    = numsim::cas::expression_holder<numsim::cas::tensor_to_scalar_expression>;

void bind_scalar(py::module_ &m, py::class_<ScalarExpr> &cls);
void bind_tensor(py::module_ &m, py::class_<TensorExpr> &cls);
void bind_tensor_to_scalar(py::module_ &m, py::class_<T2SExpr> &cls);
void bind_cross_domain(py::module_ &m,
                       py::class_<ScalarExpr> &scalar_cls,
                       py::class_<TensorExpr> &tensor_cls,
                       py::class_<T2SExpr> &t2s_cls);

#endif // NUMSIM_CAS_PYTHON_COMMON_H
