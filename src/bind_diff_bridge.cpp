// Provides visible definitions of the tag_invoke diff overloads.
//
// The numsim-cas static library internally calls diff() recursively
// (e.g. tensor_differentiation.cpp -> diff(sub_expr, arg)).  These
// calls resolve to the forward-declared tag_invoke symbols.  The
// inline definitions in tensor_diff.h / tensor_to_scalar_diff.h
// would normally satisfy them, but pybind11 compiles with
// -fvisibility-inlines-hidden, which hides inline function symbols.
//
// This TU provides non-inline definitions with default visibility
// that the linker can use to satisfy the references.

#include <numsim_cas/core/diff.h>
#include <numsim_cas/tensor/visitors/tensor_differentiation.h>
#include <numsim_cas/tensor_to_scalar/visitors/tensor_to_scalar_differentiation.h>

namespace numsim::cas {

// These match the signatures forward-declared in the differentiation
// headers and defined inline in tensor_diff.h / tensor_to_scalar_diff.h.
// We provide non-inline copies with default visibility.

__attribute__((visibility("default")))
expression_holder<tensor_expression>
tag_invoke(detail::diff_fn, std::type_identity<tensor_expression>,
           std::type_identity<tensor_expression>,
           expression_holder<tensor_expression> const &expr,
           expression_holder<tensor_expression> const &arg) {
  tensor_differentiation d(arg);
  return d.apply(expr);
}

__attribute__((visibility("default")))
expression_holder<tensor_expression>
tag_invoke(detail::diff_fn,
           std::type_identity<tensor_to_scalar_expression>,
           std::type_identity<tensor_expression>,
           expression_holder<tensor_to_scalar_expression> const &expr,
           expression_holder<tensor_expression> const &arg) {
  tensor_to_scalar_differentiation d(arg);
  return d.apply(expr);
}

} // namespace numsim::cas
