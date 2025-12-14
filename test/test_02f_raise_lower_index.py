import pytest
import sympy as sp
from dgeom.sym import *

# ==========================================
# 測試案例開始
# ==========================================

def test_raise_lower_index_polar():
    """
    測試案例：極座標 (2D)
    Metric: diag(1, r^2)
    """
    r, theta = sp.symbols('r theta', real=True, positive=True)
    coords = [r, theta]
    
    g_data = sp.diag(1, r**2)
    metric = MetricTensor(g_data, coords)

    # =======================================
    # 測試 1: 向量降指標 (Lowering)
    # =======================================
    # 定義逆變向量 v^i = [1, 1]
    # v^r = 1, v^theta = 1
    v_data = [1, 1]
    vec_up = GeometricTensor(v_data, coords, [1]) # index: [up]

    # 執行降指標 v_i = g_ij v^j
    # 理論值:
    # v_r     = g_rr * v^r     = 1 * 1 = 1
    # v_theta = g_thth * v^th  = r^2 * 1 = r^2
    vec_down = metric.lower_index(vec_up, pos=0)

    assert vec_down.index_config == [-1]
    assert sp.simplify(vec_down.data[0] - 1) == 0
    assert sp.simplify(vec_down.data[1] - r**2) == 0

    # =======================================
    # 測試 2: 向量升指標 (Raising) - Round Trip
    # =======================================
    # 將剛剛降下去的 v_i 升回來，應該要變回 [1, 1]
    vec_restored = metric.raise_index(vec_down, pos=0)
    
    assert vec_restored.index_config == [1]
    assert sp.simplify(vec_restored.data[0] - 1) == 0
    assert sp.simplify(vec_restored.data[1] - 1) == 0

def test_raise_lower_metric_identity():
    """
    測試案例：混合指標度規
    驗證 g^i_j (將 metric 第一個指標升起) 是否等於單位矩陣 (Identity Matrix)
    """
    r, theta = sp.symbols('r theta', real=True, positive=True)
    coords = [r, theta]
    
    g_data = sp.diag(1, r**2)
    metric = MetricTensor(g_data, coords) # g_ab, config [-1, -1]

    # 操作：將 pos=0 的指標由 -1 (協變) 改為 1 (逆變)
    # 結果應為 g^a_b
    mixed_metric = metric.raise_index(metric, pos=0)

    # 檢查 index config
    assert mixed_metric.index_config == [1, -1]

    # 檢查數值
    # g^a_c * g_cb = delta^a_b
    # 結果應該是單位矩陣 [[1, 0], [0, 1]]
    expected = sp.eye(2)
    
    rows, cols = 2, 2
    for i in range(rows):
        for j in range(cols):
            val = sp.simplify(mixed_metric.data[i, j])
            target = expected[i, j]
            assert val == target, f"Mismatch at {i},{j}: got {val}, expected {target}"

def test_raise_lower_errors():
    """測試錯誤處理"""
    x, y = sp.symbols('x y')
    
    metric = MetricTensor(sp.eye(2), [x, y])
    vec = GeometricTensor([1, 1], [x, y], [1])

    # 測試 1: 無效的 pos (超出範圍)
    with pytest.raises(IndexError):
        metric.lower_index(vec, pos=5)

if __name__ == "__main__":
    # 如果直接執行此腳本，自動執行 pytest
    import sys
    sys.exit(pytest.main(["-v", __file__]))