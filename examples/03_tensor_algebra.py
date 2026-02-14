"""Tensor symbolic expressions: construction, algebra, and projectors."""

import numsim_cas as cas

dim = 3
rank = 2

# --- Tensor variables ---
X = cas.tensor_variable("X", dim=dim, rank=rank)
Y = cas.tensor_variable("Y", dim=dim, rank=rank)

print("X           =", X)
print("Y           =", Y)
print(f"X.dim={X.dim}, X.rank={X.rank}")
print()

# --- Special tensors ---
Z = cas.tensor_zero(dim=dim, rank=rank)
I = cas.kronecker_delta(dim=dim)
I4 = cas.identity_tensor(dim=dim, rank=4)

print("Zero tensor =", Z)
print("delta_ij    =", I)
print("I (rank-4)  =", I4)
print()

# --- Arithmetic ---
print("X + Y       =", X + Y)
print("X - Y       =", X - Y)   # note: tensor sub simplification is WIP upstream
print("-X          =", -X)
print("X + X       =", X + X)      # 2*X
print("X + Zero    =", X + Z)      # X
print()

# --- Tensor products ---
print("X * Y       =", X * Y)      # inner product (contraction)
print("X otimes Y  =", cas.otimes(X, Y))
print("X otimesu Y =", cas.otimesu(X, Y))
print("X otimesl Y =", cas.otimesl(X, Y))
print()

# --- Tensor functions ---
print("dev(X)      =", cas.dev(X))
print("inv(X)      =", cas.inv(X))
print("trans(X)    =", cas.trans(X))
print("pow(X, 2)   =", cas.pow(X, 2))
print()

# --- Scalar * Tensor ---
a = cas.variable("a")
print("a * X       =", a * X)
print("X / a       =", X / a)
print("2 * X       =", 2 * X)
print("X / 3       =", X / 3)
print()

# --- Projectors (rank-4 tensors) ---
print("P_sym(3)    =", cas.P_sym(dim))
print("P_skew(3)   =", cas.P_skew(dim))
print("P_vol(3)    =", cas.P_vol(dim))
print("P_devi(3)   =", cas.P_devi(dim))
