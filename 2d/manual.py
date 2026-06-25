import numpy as np
from dolfinx import mesh, fem, io
from mpi4py import MPI
import basix.ufl

from lib import right_side_cp_to_whole, cp_to_density_light, solve_helmholtz

# 基本パラメータ
f = 300.0
omega = 2 * np.pi * f
c = 343.0 # 音速
k_val = omega / c

domain = mesh.create_rectangle(MPI.COMM_WORLD, points=[[0.0, 0.0], [100.0, 100.0]], n=[500, 500])

sub_element = basix.ufl.element("Lagrange", domain.topology.cell_name(), 1)
element = basix.ufl.blocked_element(sub_element, shape=(2,))

V = fem.functionspace(domain, element)
V_rho = fem.functionspace(domain, sub_element)

xdmf = io.XDMFFile(domain.comm, "dist/test_output.xdmf", "w")
xdmf.write_mesh(domain)

rho_function = fem.Function(V_rho)
rho_function.name = "Density"

right_side_cp = np.array([
    [40.0, 20.0],
    [40.0, 50.0],
    [30.0, 50.0],
    [40.0, 80.0],
])

full_cp = right_side_cp_to_whole(right_side_cp, origin_coord=[50.0, 0.0])

rho_function.x.array[:] = cp_to_density_light(full_cp, V_rho, path_width=1)

def open_boundaries(x):
    return (np.isclose(x[0], 0.0) | np.isclose(x[0], 100.0) |
            np.isclose(x[1], 0.0) | np.isclose(x[1], 100.0))


f_source = np.array([50.0, 50.0])
u_sol = solve_helmholtz(V, rho_function, k_val, f_source, open_boundaries)
u_sol.name = "Solution"

xdmf.write_function(rho_function, 0.0)
xdmf.write_function(u_sol, 0.0)
xdmf.close()
