"""Tensor-to-scalar expressions: trace, determinant, norm, and algebra."""

import numsim_cas as cas

dim = 3
X = cas.tensor_variable("X", dim=dim, rank=2)
Y = cas.tensor_variable("Y", dim=dim, rank=2)

# --- Construction from tensors ---
print("trace(X)    =", cas.trace(X))
print("det(X)      =", cas.det(X))
print("norm(X)     =", cas.norm(X))
print("dot(X)      =", cas.dot(X))
print()

# --- Algebra on tensor-to-scalar expressions ---
trX = cas.trace(X)
trY = cas.trace(Y)
detX = cas.det(X)

print("tr(X) + tr(Y)  =", trX + trY)
print("tr(X) - tr(Y)  =", trX - trY)
print("tr(X) * tr(Y)  =", trX * trY)
print("tr(X) / tr(Y)  =", trX / trY)
print("-tr(X)          =", -trX)
print("tr(X) ** 2      =", trX ** 2)
print()

# --- Mixed with numerics ---
print("tr(X) + 1       =", trX + 1)
print("2 * tr(X)       =", 2 * trX)
print("tr(X) / 3       =", trX / 3)
print()

# --- Functions on t2s expressions ---
print("log(det(X))     =", cas.log(detX))
print("pow(tr(X), 3)   =", cas.pow(trX, 3))
print()

# --- Compound expressions ---
# A common invariant: I1 = tr(X), I2 = 0.5*(tr(X)^2 - tr(X*X)), I3 = det(X)
I1 = cas.trace(X)
I3 = cas.det(X)

print("I1 = tr(X)      =", I1)
print("I3 = det(X)     =", I3)
print("I1^2 + I3       =", I1**2 + I3)
