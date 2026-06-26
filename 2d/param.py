import numpy as np

# 基本パラメータ
f = 500.0
f_source = np.array([50.0, 2.0])
omega = 2 * np.pi * f
c = 343.0 # 音速
k_val = omega / c

# メッシュ生成（粗くして高速化）
x_min, x_max = 0.0, 100.0
y_min, y_max = 0.0, 30.0
