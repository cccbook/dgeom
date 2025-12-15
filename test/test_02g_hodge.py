import pytest
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from dgeom.sym import *

# ===================================================================
# 霍奇星算子 hodge*
# ===================================================================

def test_hodge_flat_sharp_inversion():
    """
    ### 🧪 驗證：指標升降的可逆性 (Flat vs Sharp)
    驗證 HodgeMetric 是否正確實作了指標升降。
    """
    # 1. 準備度規與向量
    tm = euclidean_metric()
    x, y, z = tm.coords
    
    # 建立 HodgeMetric 介面 (若 MetricTensor 已實作 flat/sharp 可直接用，這裡假設用 HodgeWrapper)
    # 若 dgeom.sym 有直接導出 HodgeMetric，則使用它
    h_metric = HodgeMetric(tm.data, tm.coords)

    # V = x^2 ∂x + y ∂y + cos(z) ∂z
    V = TangentVector([x**2, y, sp.cos(z)], tm.coords, name='V') 
    
    # 2. 執行升降運算
    V_flat = h_metric.flat(V)        # Vector -> 1-Form (降)
    V_restored = h_metric.sharp(V_flat) # 1-Form -> Vector (升)
    
    # 3. 驗證逆運算 (V_restored == V)
    # 檢查數據差異是否為 0
    diff_data = V_restored.data - V.data
    
    for val in np.array(diff_data).flatten():
        assert sp.simplify(val) == 0

