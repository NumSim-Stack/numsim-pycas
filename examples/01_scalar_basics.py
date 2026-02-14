"""Scalar symbolic expressions: variables, constants, and algebraic simplification."""

import numsim_cas as cas

# --- Variables and constants ---
x = cas.variable("x")
y = cas.variable("y")
c = cas.constant(3)

print("x       =", x)
print("y       =", y)
print("c       =", c)
print("ZERO    =", cas.ZERO)
print("ONE     =", cas.ONE)
print()

# --- Arithmetic operators ---
print("x + y   =", x + y)
print("x - y   =", x - y)
print("x * y   =", x * y)
print("x / y   =", x / y)
print("-x      =", -x)
print("x ** 3  =", x ** 3)
print()

# --- Automatic simplification ---
print("x + x       =", x + x)          # 2*x
print("x * x       =", x * x)          # pow(x,2)
print("x + 0       =", x + cas.ZERO)   # x
print("x * 1       =", x * cas.ONE)    # x
print("x * 0       =", x * cas.ZERO)   # 0
print("x - x       =", x - x)          # 0
print("3 * x + 2   =", 3 * x + 2)
print()

# --- Math functions ---
print("sin(x)  =", cas.sin(x))
print("cos(x)  =", cas.cos(x))
print("exp(x)  =", cas.exp(x))
print("log(x)  =", cas.log(x))
print("sqrt(x) =", cas.sqrt(x))
print("abs(x)  =", cas.abs(x))
print()

# --- Building compound expressions ---
f = x**2 + 2*x*y + y**2
print("f = x^2 + 2xy + y^2 =", f)
