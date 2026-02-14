"""Symbolic differentiation and numeric evaluation of scalar expressions."""

import numsim_cas as cas

x, y = cas.variables("x", "y")

# --- Differentiation ---
f = x**2 + 2*x*y + y**2
print("f           =", f)
print("df/dx       =", cas.diff(f, x))
print("df/dy       =", cas.diff(f, y))
print()

# Chain rule through trig functions
g = cas.sin(x**2)
print("g           =", g)
print("dg/dx       =", cas.diff(g, x))
print()

# Product rule
h = x * cas.exp(x)
print("h           =", h)
print("dh/dx       =", cas.diff(h, x))
print()

# Higher-order derivatives
print("d2f/dx2     =", cas.diff(cas.diff(f, x), x))
print()

# --- Evaluation ---
expr = x**2 + 3*y
print("expr        =", expr)
print("expr(1, 2)  =", cas.evaluate(expr, {x: 1.0, y: 2.0}))  # 1 + 6 = 7
print("expr(3, -1) =", cas.evaluate(expr, {x: 3.0, y: -1.0})) # 9 - 3 = 6
print()

# Evaluate a derivative
df = cas.diff(expr, x)  # 2*x
print("df/dx       =", df)
print("df/dx(5, 0) =", cas.evaluate(df, {x: 5.0, y: 0.0}))  # 10

# Evaluate trig expression
import math
val = cas.evaluate(cas.sin(x), {x: math.pi / 2})
print(f"sin(pi/2)   = {val}")
