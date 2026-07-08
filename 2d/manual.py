import numpy as np
from dolfinx import mesh, fem, io
from mpi4py import MPI
import sys
import basix.ufl

from lib import right_side_cp_to_whole, cp_to_density_light, solve_helmholtz
from param import f, f_source, k_val, x_min, x_max, y_min, y_max, nx, ny

domain = mesh.create_rectangle(MPI.COMM_WORLD, points=[[x_min, y_min], [x_max, y_max]], n=[nx, ny])

sub_element = basix.ufl.element("Lagrange", domain.topology.cell_name(), 1)
element = basix.ufl.blocked_element(sub_element, shape=(2,))

V = fem.functionspace(domain, element)
V_rho = fem.functionspace(domain, sub_element)

xdmf = io.XDMFFile(domain.comm, "dist/test_output.xdmf", "w")
xdmf.write_mesh(domain)

rho_function = fem.Function(V_rho)
rho_function.name = "Density"

right_side_cp = np.array([
    [15.0, 2.0],
    [15.0, 13.0],
    [11.0, 14.0],
    [6.0, 15.0],
])

full_cp = right_side_cp_to_whole(right_side_cp, origin_coord=[(x_min+x_max)/2, y_min])

rho_function.x.array[:] = cp_to_density_light(full_cp, V_rho, path_width=1)

def open_boundaries(x):
    return (np.isclose(x[0], x_min) | np.isclose(x[0], x_max) |
            np.isclose(x[1], y_min) | np.isclose(x[1], y_max))


f_source = np.array([50.0, 50.0])
u_sol = solve_helmholtz(V, rho_function, k_val, f_source, open_boundaries)
u_sol.name = "Solution"

V_score = fem.functionspace(domain, ("DG", 0))
step_score = fem.Function(V_score, name="Step_Score")
step_score.x.array[:] = 13.0

xdmf.write_function(rho_function, 0.0)
xdmf.write_function(u_sol, 0.0)
xdmf.write_function(step_score, 0.0)
xdmf.close()

MPI.Finalize()
sys.exit()
