import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
import ufl

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
    amplitude: float = 1.0,       # 音源の振幅 (音量)
    c: float = 343.0
) -> float:
    
    comm = domain.comm
    omega = 2.0 * np.pi * freq
    k = omega / c
    
    V = fem.functionspace(domain, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    
    # --- 1. 音源振幅 (amplitude) を組み込んだガウス音源 ---
    x = ufl.SpatialCoordinate(domain)
    x0, y0 = source_pos
    sigma = 0.02
    r_sq = (x[0] - x0)**2 + (x[1] - y0)**2
    # 振幅 amplitude を掛け合わせる
    source_expr = amplitude * ufl.exp(-r_sq / (2.0 * sigma**2))
    
    dx = ufl.Measure("dx", domain=domain)
    a = (ufl.inner(ufl.grad(u), ufl.grad(v)) - (k**2) * ufl.inner(u, v)) * dx
    L = source_expr * v * dx
    
    problem = fem.petsc.LinearProblem(a, L, bcs=[])
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
        
    # 聴覚補正後の等価音圧 [Pa] に逆換算
    # p_rms_perceptual = p_ref * (10.0 ** (spl_dBA / 20.0))
        
    return float(spl_dBA) # 聴覚補正後の騒音レベル[dB]
