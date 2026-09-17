from mpi4py import MPI
from dolfinx import mesh

# ドメインメッシュを作成
comm = MPI.COMM_WORLD # MPIコミュニケータ
p_min = (-50.0, 0.0) # 領域の左下と右上の座標
p_max = (50.0, 20.0)
cell_type = mesh.CellType.triangle # 分割の方法

x_coords = [-15.0, -12.0, -9.0, -6.0, -3.0,  0.0,  3.0,  6.0,  9.0, 12.0, 15.0]
y_init   = [  2.0,  10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,  2.0]

test_freqs = [125.0, 250.0, 500.0, 1000.0, 2000.0]
test_amplitude = 80.0 # [dB]


output_dir = "./dist"
