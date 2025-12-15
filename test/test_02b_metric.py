import pytest
import sympy as sp
import numpy as np
from dgeom.sym import *

# ===================================================================
# 測試 1: 度規性質驗證 (Validation)
# ===================================================================

def test_metric_validation_singular():
    """
    測試：初始化退化 (Singular) 的度規應拋出錯誤。
    行列式為 0 的度規無法定義逆度規，幾何結構不存在。
    """
    x, y = sp.symbols('x y')
    coords = [x, y]
    
    # 定義一個行列式為 0 的矩陣 (rank deficient)
    # diag(1, 0) -> det = 0
    g_data = [[1, 0], [0, 0]]
    
    # 預期拋出 ValueError，訊息包含 "退化" 或 "Singular"
    with pytest.raises(ValueError, match="退化|Singular"):
        MetricTensor(g_data, coords)

def test_metric_validation_nonsquare():
    """
    測試：非方陣數據不能作為度規。
    """
    x, y = sp.symbols('x y')
    coords = [x, y]
    g_data = [[1, 0, 0], [0, 1, 0]] # 2x3

    # [修正] 預期錯誤訊息改為匹配父類別 GeometricTensor 拋出的訊息
    # 原本是 match="方陣"，現在改為 match="維度" 或 "不一致"
    with pytest.raises(ValueError, match="維度"):
        MetricTensor(g_data, coords)

# ===================================================================
# 測試 2: 歐幾里得空間的測量 (Euclidean Measurements)
# ===================================================================

def test_euclidean_norm_and_angle():
    """
    測試：在平直空間計算向量長度與夾角。
    """
    tm = euclidean_metric() # x, y, z
    
    # 定義兩個向量 (使用 list，MetricTensor 會自動轉為逆變張量)
    # u = 3 dx + 4 dy
    u = [3, 4, 0]
    # v = 0 dx + 5 dy
    v = [0, 5, 0]
    
    # 1. 測試長度 (Norm)
    # ||u|| = sqrt(3^2 + 4^2) = 5
    assert sp.simplify(tm.norm(u) - 5) == 0
    
    # 2. 測試內積 (Inner Product)
    # <u, v> = 3*0 + 4*5 + 0*0 = 20
    assert sp.simplify(tm.inner_product(u, v) - 20) == 0
    
    # 3. 測試夾角 (Angle)
    # cos(theta) = <u,v> / (|u||v|) = 20 / (5 * 5) = 4/5
    # theta = acos(4/5)
    angle_calc = tm.angle(u, v)
    angle_expected = sp.acos(sp.Rational(4, 5))
    
    assert sp.simplify(angle_calc - angle_expected) == 0

def test_euclidean_orthogonality():
    """
    測試：正交向量的內積應為 0，夾角應為 pi/2。
    """
    tm = euclidean_metric()
    
    v1 = [1, 0, 0] # x 軸
    v2 = [0, 1, 0] # y 軸
    
    # 內積應為 0
    assert tm.inner_product(v1, v2) == 0
    
    # 夾角應為 90度 (pi/2)
    assert tm.angle(v1, v2) == sp.pi / 2

# ===================================================================
# 測試 3: 彎曲空間的測量 (Curved Space Measurements)
# ===================================================================

def test_spherical_norm():
    """
    測試：球坐標下的向量長度。
    基底向量 e_theta 的長度不是 1，而是 r。
    """
    tm = spherical_metric()
    r, theta, phi = tm.coords
    
    # 定義單位角向量 (Coordinate Basis Vector)
    # V = d/d_theta = [0, 1, 0]
    v_theta = [0, 1, 0]
    
    # 計算範數
    # g_theta_theta = r^2
    # <v, v> = r^2 * 1 * 1 = r^2
    # ||v|| = sqrt(r^2) = r
    norm = tm.norm(v_theta)
    
    assert sp.simplify(norm - r) == 0

def test_spherical_orthogonality():
    """
    測試：球坐標系是正交坐標系 (Orthogonal Coordinates)。
    雖然空間彎曲，但基底向量彼此垂直。
    """
    tm = spherical_metric()
    
    v_r = [1, 0, 0]     # 徑向
    v_theta = [0, 1, 0] # 切向
    
    # 檢查是否垂直
    angle = tm.angle(v_r, v_theta)
    
    assert angle == sp.pi / 2

# ===================================================================
# 測試 4: 閔可夫斯基空間 (Minkowski / Relativity)
# ===================================================================

def test_minkowski_light_cone():
    """
    測試：狹義相對論中的類光向量 (Light-like Vector)。
    度規: diag(1, -1, -1, -1)
    """
    t, x, y, z = sp.symbols('t x y z')
    coords = [t, x, y, z]
    # (+, -, -, -)
    g_data = sp.diag(1, -1, -1, -1)
    
    tm = MetricTensor(g_data, coords)
    
    # 定義光速運動的向量 (c=1)
    # v = (1, 1, 0, 0) -> dt=1, dx=1
    light_vec = [1, 1, 0, 0]
    
    # 1. 內積自乘 (Interval)
    # s^2 = 1*1^2 - 1*1^2 = 0
    interval = tm.inner_product(light_vec, light_vec)
    assert interval == 0
    
    # 2. 範數 (Norm)
    # sqrt(0) = 0
    assert tm.norm(light_vec) == 0
    
    # 3. 測試夾角報錯 (零向量無法計算夾角)
    # 分母為 0，應拋出 ValueError
    other_vec = [1, 0, 0, 0]
    with pytest.raises(ValueError, match="無法計算"):
        tm.angle(light_vec, other_vec)

# ===================================================================
# 測試 5: 輸入檢查 (Input Validation)
# ===================================================================

def test_inner_product_covariant_error():
    """
    測試：若傳入協變向量 (Covariant, 下標) 進行內積，應報錯。
    內積定義為 g_ij u^i v^j，輸入必須是逆變向量。
    """
    tm = euclidean_metric()
    x, y, z = tm.coords
    
    # 建立一個協變向量 (Type=[-1])
    # 例如梯度的結果
    v_cov = GeometricTensor([1, 0, 0], [x, y, z], [-1])
    v_contra = GeometricTensor([1, 0, 0], [x, y, z], [1])
    
    # 嘗試計算內積
    with pytest.raises(ValueError, match="逆變向量"):
        tm.inner_product(v_cov, v_contra)


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
    import sys
    sys.exit(pytest.main(["-v", __file__]))