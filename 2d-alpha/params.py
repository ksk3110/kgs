from mpi4py import MPI
from dolfinx import mesh

# ドメインメッシュを作成
comm = MPI.COMM_WORLD # MPIコミュニケータ
points = ((-50.0, 0.0), (50.0, 20.0)) # 領域の左下と右上の座標
cell_type = mesh.CellType.triangle # 分割の方法

test_freqs = [125.0, 250.0, 500.0, 1000.0, 2000.0]
test_amplitude = 80.0 # [dB]
