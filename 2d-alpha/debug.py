from scipy.optimize import minimize

import lib
import params
import os
import numpy as np

step_counter = 0

def objective(y_vars):
    global step_counter

    control_points = list(zip(
        [-x for x in params.x_coords[::-1]] + params.x_coords,
        [*[y for y in y_vars[::-1]], *y_vars]
    ))

    # ステップごとの出力ファイル名 (例: ./optimization_steps/step_000.xdmf)
    filename = os.path.join(params.output_dir, f"step_{step_counter:03d}.xdmf")

    try:
        # 1. メッシュ生成
        domain, cell_tags, facet_tags = lib.rho_fields_from_controls(
            control_points=control_points,
            lc_domain=2.0,
            lc_wall=0.5,
        )

        # 2. ヘルムホルツ解析 ＆ XDMFファイル出力
        J = lib.solve_helmholtz_and_evaluate_perceptual_rms(
            domain=domain,
            facet_tags=facet_tags,
            cell_tags=cell_tags,
            freq=params.test_freqs[0],
            source_pos=(0.0, 1.0),
            amplitude_db=params.test_amplitude,
            output_filename=filename,  # 連番出力
            step=step_counter
        )

        print(f"Step {step_counter:03d} | y: {np.round(y_vars, 2)} -> J(dBA): {J:.4f}")
        step_counter += 1
        return J

    except Exception as e:
        print(f"Step {step_counter:03d} Failed: {e}")
        return 1e6

bounds = [(2.0, 12.0) for _ in params.y_init]
res = minimize(
    objective,
    x0=params.y_init,
    method="COBYLA",
    bounds=bounds,
    options={"maxiter": 50}
)
