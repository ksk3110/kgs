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

res = 5 # 単位長さあたりのメッシュの解像度
nx = int((x_max - x_min) * res)
ny = int((y_max - y_min) * res)

# 設計可能領域
def desinable(coord):
  return (
      (35.0 <= coord[0] <= 55.0) and
      (1.0 <= coord[1] <= 20.0)
  )
