import basix
from basix.ufl import element
from ufl import Coefficient, FunctionSpace, Mesh, TestFunction, grad, dx, inner

P = 4  # Degree of polynomial basis
Q = 5  # Number of quadrature points

# Define mesh and finite element
coord_element = element("Lagrange", "hexahedron", 1, shape=(3,))
mesh = Mesh(coord_element)
e = element(
    basix.ElementFamily.P,
    basix.CellType.hexahedron,
    P,
    basix.LagrangeVariant.gll_warped,
)
e_DG = element(
    basix.ElementFamily.P,
    basix.CellType.hexahedron,
    0,
    basix.LagrangeVariant.gll_warped,
    basix.DPCVariant.unset,
    True,
)

# Define function spaces
V = FunctionSpace(mesh, e)
V_DG = FunctionSpace(mesh, e_DG)

c0 = Coefficient(V_DG)

u = Coefficient(V)
v = TestFunction(V)

# Map from quadrature points to basix quadrature degree
qdegree = {3: 4, 4: 5, 5: 6, 6: 8, 7: 10, 8: 12, 9: 14, 10: 16}
md = {"quadrature_rule": "GLL", "quadrature_degree": qdegree[Q]}

# Define operators
m = inner(c0 * u, v) * dx(metadata=md)
s = inner(c0 * grad(u), grad(v)) * dx(metadata=md)

f0 = Coefficient(V)
f1 = Coefficient(V)

E = inner(f1 - f0, f1 - f0) * dx

forms = [m, s, E]
