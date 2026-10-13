from mpi4py import MPI
from dolfinx import mesh
import numpy as np

# ドメインメッシュを作成
comm = MPI.COMM_WORLD # MPIコミュニケータ
p_min = (-50.0, 0.0) # 領域の左下と右上の座標
p_max = (50.0, 20.0)
cell_type = mesh.CellType.triangle # 分割の方法

x_coords = [ 0.0,  3.0,  6.0,  9.0, 12.0, 15.0]
y_init   = [10.0, 10.0, 10.0, 10.0, 10.0,  2.0]

test_freqs = [125.0, 250.0, 500.0, 1000.0, 2000.0]
test_amplitude = 80.0 # [dB]

def target_region(x):
    return (16.0 <= x[0]) & (x[0] <= 49.0) & (1.0 <= x[1]) & (x[1] <= 10.0)

output_dir = "./dist"
