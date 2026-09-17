from mpi4py import MPI
from dolfinx import mesh

# ドメインメッシュを作成
def create_domain() -> mesh.Mesh:
  comm = MPI.COMM_WORLD # MPIコミュニケータ
  points = ((-50.0, 0.0), (50, 30)) # 領域の左下と右上の座標
  cell_type = dolfinx.mesh.CellType.triangle # 分割の方法
  domain = mesh.create_rectangle(
    comm,
    points,
    cell_type
  )

  return domain
