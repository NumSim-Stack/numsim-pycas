"""Strain energy functions for hyperelastic material models.

Demonstrates how to build common continuum-mechanics strain energy
densities using tensor, tensor-to-scalar, and scalar expressions.
"""

import numsim_cas as cas

dim = 3

# --- Kinematic quantities ---
C = cas.tensor_variable("C", dim=dim, rank=2)   # right Cauchy-Green tensor
cas.assume_symmetric(C)                          # C is symmetric by definition
I = cas.kronecker_delta(dim=dim)                 # 2nd-order identity

# Principal invariants of C
I1 = cas.trace(C)           # I1 = tr(C)
I2 = (I1**2 - cas.dot(C)) / 2  # I2 = (tr(C)^2 - tr(C^2)) / 2  (dot = A:A)
I3 = cas.det(C)             # I3 = det(C)
J  = cas.pow(I3, 0.5)       # J  = sqrt(det(C))

print("=== Principal invariants of C ===")
print("I1 = tr(C)                        =", I1)
print("I2 = (tr(C)^2 - C:C) / 2         =", I2)
print("I3 = det(C)                       =", I3)
print("J  = det(C)^(1/2)                 =", J)
print()

# =====================================================================
# 1. Saint-Venant-Kirchhoff
#    W = lambda/2 * tr(E)^2 + mu * E:E
#    where E = (C - I) / 2 is the Green-Lagrange strain
# =====================================================================
lam = cas.variable("lambda")
mu  = cas.variable("mu")

E = (C - I) / 2                     # Green-Lagrange strain tensor
trE  = cas.trace(E)
dotE = cas.dot(E)                    # E:E = tr(E^T E)

W_svk = lam / 2 * trE**2 + mu * dotE

print("=== Saint-Venant-Kirchhoff ===")
print("E  = (C - I) / 2")
print("W  = lam/2 * tr(E)^2 + mu * (E:E)")
print("W  =", W_svk)
print()

# =====================================================================
# 2. Neo-Hookean (compressible, Simo-Pister form)
#    W = mu/2 * (I1 - 3) - mu * ln(J) + lambda/2 * ln(J)^2
# =====================================================================
lnJ = cas.log(I3) / 2               # ln(J) = ln(det(C)) / 2

W_nh = mu / 2 * (I1 - 3) - mu * lnJ + lam / 2 * lnJ**2

print("=== Neo-Hookean (Simo-Pister) ===")
print("W  = mu/2*(I1 - 3) - mu*ln(J) + lam/2*ln(J)^2")
print("W  =", W_nh)
print()

# =====================================================================
# 3. Mooney-Rivlin
#    W = c1 * (I1 - 3) + c2 * (I2 - 3)
# =====================================================================
c1 = cas.variable("c1")
c2 = cas.variable("c2")

W_mr = c1 * (I1 - 3) + c2 * (I2 - 3)

print("=== Mooney-Rivlin ===")
print("W  = c1*(I1 - 3) + c2*(I2 - 3)")
print("W  =", W_mr)
print()

# =====================================================================
# 4. Isochoric-volumetric split (Flory decomposition)
#    Cbar = J^{-2/3} * C   (modified deformation tensor)
#    Ibar1 = tr(Cbar) = J^{-2/3} * I1
#    W_iso = mu/2 * (Ibar1 - 3)
#    W_vol = kappa/2 * (J - 1)^2
#    W     = W_iso + W_vol
# =====================================================================
kappa = cas.variable("kappa")

Jm23   = cas.pow(I3, -1.0 / 3.0)    # J^{-2/3} = det(C)^{-1/3}
Ibar1  = Jm23 * I1                   # modified first invariant

W_iso = mu / 2 * (Ibar1 - 3)
W_vol = kappa / 2 * (J - 1)**2
W_flory = W_iso + W_vol

print("=== Isochoric-volumetric split ===")
print("J^{-2/3}  =", Jm23)
print("Ibar1     =", Ibar1)
print("W_iso     =", W_iso)
print("W_vol     =", W_vol)
print("W         =", W_flory)
print()

# =====================================================================
# 5a. Tensor space assumptions: sym(C) → C, skew(C) → 0
# =====================================================================
print("=== Tensor space assumptions ===")
print("C is symmetric:", cas.is_symmetric(C))
print("sym(C) → C:   ", cas.sym(C))                 # should print "C"
print("skew(C) → 0:  ", cas.skew(C))                # should print zero
print()

# =====================================================================
# 5. Using projection functions (dev, vol, sym)
#    Demonstrate the deviatoric/volumetric decomposition on strain
# =====================================================================
print("=== Strain decomposition with projections ===")
print("dev(C)    =", cas.dev(C))
print("vol(C)    =", cas.vol(C))
print("sym(C)    =", cas.sym(C))
print()

# Strain energy in terms of deviatoric and volumetric parts
dev_E = cas.dev(E)
vol_E = cas.vol(E)
print("dev(E)    =", dev_E)
print("vol(E)    =", vol_E)

# W = mu * dev(E):dev(E) + kappa/2 * tr(E)^2
dot_dev_E = cas.dot(dev_E)
W_split = mu * dot_dev_E + kappa / 2 * trE**2

print("W  = mu * dev(E):dev(E) + kappa/2 * tr(E)^2")
print("W  =", W_split)
print()

# =====================================================================
# 6. Tensor differentiation basics
#    dC/dC = I4    d(tr(C))/dC = I    d(C:C)/dC = 2*C
# =====================================================================
print("=== Tensor differentiation basics ===")

dCdC = cas.diff(C, C)
print("dC/dC           =", dCdC)
print(f"   rank = {dCdC.rank}")          # rank 4

dtrC = cas.diff(I1, C)
print("d(tr(C))/dC     =", dtrC)
print(f"   rank = {dtrC.rank}")          # rank 2

dotC = cas.dot(C)
ddotC = cas.diff(dotC, C)
print("d(C:C)/dC       =", ddotC)       # 2*C

dtrC2 = cas.diff(I1**2, C)
print("d(tr(C)^2)/dC   =", dtrC2)       # 2*tr(C)*I
print()

# =====================================================================
# 7. Stress from Saint-Venant-Kirchhoff (polynomial → clean derivatives)
#    S = 2 * dW/dC = lambda * tr(E) * I + 2 * mu * E
# =====================================================================
print("=== Saint-Venant-Kirchhoff: stress via dW/dC ===")

S_svk = 2 * cas.diff(W_svk, C)
print("S  = 2*dW/dC  =", S_svk)
print()

# =====================================================================
# 8. Material tangent via second derivative
#    CC = 2 * dS/dC = 4 * d^2W/dC^2
# =====================================================================
print("=== Saint-Venant-Kirchhoff: tangent via d^2W/dC^2 ===")

CC_svk = 2 * cas.diff(S_svk, C)
print("CC = 2*dS/dC  =", CC_svk)
print(f"   rank = {CC_svk.rank}")        # rank 4
print()

# =====================================================================
# 9. Scalar differentiation of a 1D strain energy
# =====================================================================
print("=== 1D scalar strain energy + differentiation ===")
eps = cas.variable("eps")      # 1D strain
E_mod = cas.variable("E")      # Young's modulus

W_1d = E_mod / 2 * eps**2
sigma = cas.diff(W_1d, eps)    # stress = dW/d(eps)
dsigma = cas.diff(sigma, eps)  # tangent = d^2W/d(eps)^2

print("W(eps)    =", W_1d)
print("sigma     = dW/deps  =", sigma)
print("C_tangent = d2W/deps2 =", dsigma)
