import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, io
from dolfinx.io import gmsh as fenics_gmsh
from dolfinx.fem import petsc
import gmsh
import ufl
import params

def calculate_a_weighting(freq: float) -> float:
    """周波数 f [Hz] におけるA特性補正量 [dB] を計算する"""
    f2 = freq**2
    num = 12194.0**2 * f2**2
    den = (f2 + 20.6**2) * np.sqrt((f2 + 107.7**2) * (f2 + 737.9**2)) * (f2 + 12194.0**2)
    ra = num / den
    return 2.0 + 20.0 * np.log10(ra)

def solve_helmholtz_and_evaluate_perceptual_rms(
    domain: mesh.Mesh,
    freq: float,                  # 周波数 [Hz]
    source_pos: tuple,            # 音源位置 (x0, y0)
    amplitude_db: float = 96.0,       # 音源の振幅 (音量)
    c: float = 343.0,
    output_filename: str = None,
    step: int = 0
) -> float:

    comm = domain.comm
    omega = 2.0 * np.pi * freq
    k = omega / c

    p0 = 2.0e-5
    p_amp = p0 * (10.0 ** (amplitude_db / 20.0))

    V = fem.functionspace(domain, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

    # --- 1. 音源振幅 (amplitude) を組み込んだガウス音源 ---
    x = ufl.SpatialCoordinate(domain)
    x0, y0 = source_pos
    sigma = 1.0
    r_sq = (x[0] - x0)**2 + (x[1] - y0)**2
    # 振幅 amplitude を掛け合わせる
    source_expr = p_amp * ufl.exp(-r_sq / (2.0 * sigma**2)) # ヘルムホルツ方程式ではfとなっている

    dx = ufl.Measure("dx", domain=domain)
    a = (ufl.inner(ufl.grad(u), ufl.grad(v)) - (k**2) * ufl.inner(u, v)) * dx
    L = source_expr * v * dx

    problem = petsc.LinearProblem(
        a, L, bcs=[],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        petsc_options_prefix="helmholtz_solver_",
    )
    p_field = problem.solve()

    # --- 2. 物理的な RMS 音圧 [Pa] の計算 ---
    p_sq_form = fem.form(ufl.inner(p_field, p_field) * dx)
    vol_form = fem.form(1.0 * dx)

    total_p_sq = comm.allreduce(np.real(fem.assemble_scalar(p_sq_form)), op=MPI.SUM)
    total_vol = comm.allreduce(np.real(fem.assemble_scalar(vol_form)), op=MPI.SUM)

    p_rms_phys = np.sqrt(total_p_sq / total_vol) # 物理的な実効音圧 [Pa]

    # --- 3. 音圧レベル (SPL [dB]) および 聴覚補正 (dBA) の計算 ---
    p_ref = 2.0e-5 # 基準音圧 20 µPa
    spl_dB = 20.0 * np.log10(p_rms_phys / p_ref) if p_rms_phys > 0 else -np.inf

    a_weight_dB = calculate_a_weighting(freq)
    spl_dBA = spl_dB + a_weight_dB  # A特性補正後の音圧レベル [dBA]


    # === Paraview用に可視化 ===
    if output_filename is not None:
        V_out = fem.functionspace(domain, ("Lagrange", 1))

        # 物理音圧の絶対値 |p| [Pa]
        p_abs = fem.Function(V_out, name="Pressure_Abs_Pa")
        p_abs.interpolate(fem.Expression(
            ufl.sqrt(ufl.inner(p_field, p_field)),
            V_out.element.interpolation_points
        ))

        # 音圧レベル Lp [dB SPL]
        p_db = fem.Function(V_out, name="Pressure_Level_dB")
        p_abs_vals = p_abs.x.array
        p_db.x.array[:] = 20.0 * np.log10(np.maximum(p_abs_vals, 1e-12) / p0)

        # メッシュと音圧フィールド（Pa & dB）の書き出し
        with io.XDMFFile(domain.comm, output_filename, "w") as xdmf:
            xdmf.write_mesh(domain)
            xdmf.write_function(p_abs, step)
            xdmf.write_function(p_db, step)


    return float(spl_dBA) # 聴覚補正後の騒音レベル[dBA]

def rho_fields_from_controls(
    control_points: list,
    lc_domain: float = 2.0,    # 領域全体のメッシュサイズ (100x30に対して2.0~3.0が適正)
    lc_wall: float = 0.5,      # 壁周辺のメッシュサイズ (制御点間隔より十分小さく設定)
    comm: MPI.Comm = MPI.COMM_WORLD
):

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)

    # --- エラー回路回避用オプション ---
    # 1. 幾何交差のトレランス（許容誤差）を少し緩める
    gmsh.option.setNumber("Geometry.Tolerance", 1e-4)

    # 2. メッシュ分割アルゴリズム: 1=MeshAdapt, 6=Frontal-Delaunay (エラーに最も強い)
    gmsh.option.setNumber("Mesh.Algorithm", 6)

    # 3. 1Dメッシュ（曲線）の分解能を高めて Edge Recovery エラーを防止
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 1)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 20) # 曲線に沿って細分化

    gmsh.model.add("sound_domain_robust")

    x_min, y_min = params.p_min
    x_max, y_max = params.p_max
    width = x_max - x_min
    height = y_max - y_min

    # 1. 領域と壁の作成
    rect = gmsh.model.occ.addRectangle(x_min, y_min, 0, width, height)
    wall_pts = [gmsh.model.occ.addPoint(pt[0], pt[1], 0, lc_wall) for pt in control_points]
    wall_curve = gmsh.model.occ.addBSpline(wall_pts)

    gmsh.model.occ.synchronize()

    # 2. 領域内に壁の曲線を埋め込み
    gmsh.model.mesh.embed(1, [wall_curve], 2, rect)

    # 3. Physical Group の割り当て（タグ重複防止のため -1 を使用）
    gmsh.model.addPhysicalGroup(2, [rect], 1, name="domain")
    gmsh.model.addPhysicalGroup(1, [wall_curve], 10, name="wall")

    # 4. メッシュサイズの設定と生成
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), lc_domain)
    wall_entities = [(0, p) for p in wall_pts]
    gmsh.model.mesh.setSize(wall_entities, lc_wall)
    gmsh.model.mesh.generate(2)

    # 5. FEniCSx メッシュおよび Tags の抽出
    partitioner = mesh.create_cell_partitioner(mesh.GhostMode.none)
    mesh_data = fenics_gmsh.model_to_mesh(
        gmsh.model, comm=comm, rank=0, gdim=2, partitioner=partitioner
    )

    domain = mesh_data.mesh
    cell_tags = mesh_data.cell_tags
    facet_tags = mesh_data.facet_tags

    gmsh.finalize()
    return domain, cell_tags, facet_tags
