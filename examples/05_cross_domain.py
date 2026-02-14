"""Cross-domain operations: mixing scalar, tensor, and tensor-to-scalar expressions."""

import numsim_cas as cas

# --- Setup ---
x = cas.variable("x")
lam = cas.variable("lambda")
mu = cas.variable("mu")

dim = 3
C = cas.tensor_variable("C", dim=dim, rank=2)
E = cas.tensor_variable("E", dim=dim, rank=2)
I = cas.kronecker_delta(dim=dim)

# --- Scalar * Tensor ---
print("=== Scalar x Tensor ===")
print("lambda * C  =", lam * C)
print("C * mu      =", C * mu)
print("C / x       =", C / x)
print()

# --- Scalar + Tensor-to-scalar ---
print("=== Scalar + Tensor-to-scalar ===")
trC = cas.trace(C)
print("x + tr(C)   =", x + trC)
print("tr(C) + x   =", trC + x)
print("x * tr(C)   =", x * trC)
print("tr(C) / x   =", trC / x)
print()

# --- Building a strain energy function ---
# Neo-Hookean: W = mu/2 * (tr(C) - 3) + lambda/2 * (log(det(C)))^2
print("=== Neo-Hookean strain energy (symbolic) ===")
trC = cas.trace(C)
detC = cas.det(C)
logJ = cas.log(detC)

term1 = mu * trC
term2 = lam * logJ**2
W = term1 + term2
print("mu*tr(C)                =", term1)
print("lambda*log(det(C))^2    =", term2)
print("W = mu*tr(C) + lam*(...) =", W)
print()

# --- Combining trace expressions ---
print("=== Tensor invariant combinations ===")
trE = cas.trace(E)
detE = cas.det(E)
print("tr(C) + tr(E)           =", trC + trE)
print("tr(C) * det(E)          =", trC * detE)
print("det(C) / det(E)         =", detC / detE)
