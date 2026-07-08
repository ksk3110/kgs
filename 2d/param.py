import numpy as np

# 基本パラメータ
f = 500.0
f_source = np.array([50.0, 2.0])
omega = 2 * np.pi * f
c = 343.0 # 音速
k_val = omega / c

# メッシュ
x_min, x_max = 0.0, 100.0
y_min, y_max = 0.0, 30.0

res = 5 # 単位長さあたりのメッシュの解像度
nx = int((x_max - x_min) * res)
ny = int((y_max - y_min) * res)

# 設計可能領域
def desinable(coord):
    return (
        (35.0 <= coord[0] <= 55.0) and
        (1.0 <= coord[1] <= 20.0)
    )

# 目的領域
ox_min, ox_max = 65.0, x_max
oy_min, oy_max = y_min, y_max

initial_rcp = np.array([
  [15.0, 2.0],
  [15.0, 8.0],
  [11.0, 11.0],
  [0.0, 13.0]
])
