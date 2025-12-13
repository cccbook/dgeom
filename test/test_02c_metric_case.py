import pytest
import sympy as sp
import numpy as np
from sympy import Function, symbols, sin, cos, pi, simplify
from dgeom.sym import TensorMetric, GeometricTensor

# ==========================================
# Fixtures: 準備常用的幾何場景
# ==========================================

@pytest.fixture
def polar_metric_setup():
    """
    場景 1: 2D 極座標 (平直空間)
    ds^2 = dr^2 + r^2 dtheta^2
    """
    r, theta = symbols('r theta', real=True, positive=True)
    coords = [r, theta]
    # g_ij
    data = [[1, 0], [0, r**2]]
    metric = TensorMetric(data, coords)
    return metric, coords

@pytest.fixture
def sphere_metric_setup():
    """
    場景 2: 2D 單位球面 (彎曲空間, 正曲率)
    ds^2 = dtheta^2 + sin^2(theta) dphi^2
    """
    theta, phi = symbols('theta phi', real=True)
    # 限制 theta 範圍以避免 0 或 pi 的奇異點，幫助 simplify
    # 注意: 在 sympy 運算中，設定 positive=True 有助於簡化 sqrt(sin^2)
    theta = symbols('theta', real=True, positive=True) 
    
    coords = [theta, phi]
    data = [[1, 0], [0, sin(theta)**2]]
    metric = TensorMetric(data, coords)
    return metric, coords

# ==========================================
# 測試案例
# ==========================================

def test_initialization_checks(polar_metric_setup):
    """測試初始化時的維度檢查"""
    metric, coords = polar_metric_setup
    
    # 1. 檢查是否正確鎖定為協變 Rank 2 ([-1, -1])
    assert metric.index_config == [-1, -1]
    
    # 2. 測試非方陣 (或維度不符) 報錯
    # GeometricTensor 父類別會先檢查 "數據維度" 與 "座標維度" 是否一致
    # 輸入 (2, 3) 的數據，但座標只有 2 維，所以會觸發父類別的 ValueError
    
    # 【修正】: 預期錯誤訊息改為匹配父類別的訊息 (只要包含 "維度" 關鍵字即可)
    with pytest.raises(ValueError, match="維度"):
        TensorMetric([[1, 0, 0], [0, 1, 0]], coords) # 2x3 matrix

def test_inverse_metric(polar_metric_setup):
    """測試逆度規計算"""
    metric, (r, theta) = polar_metric_setup
    g_inv = metric.inverse()
    
    # g^ij 應該是 diag(1, 1/r^2)
    # 檢查組態
    assert g_inv.index_config == [1, 1]
    
    # 檢查數值
    assert simplify(g_inv.data[0, 0] - 1) == 0
    assert simplify(g_inv.data[1, 1] - r**(-2)) == 0
    assert g_inv.data[0, 1] == 0

def test_christoffel_polar(polar_metric_setup):
    """
    測試極座標的克里斯多福符號
    標準結果:
    Gamma^r_theta,theta = -r
    Gamma^theta_r,theta = 1/r
    Gamma^theta_theta,r = 1/r
    其餘為 0
    """
    metric, (r, theta) = polar_metric_setup
    gamma = metric.christoffel_symbols()
    
    # Gamma indices: [k, i, j] (Upper, Lower, Lower)
    # 0 -> r, 1 -> theta
    
    # 驗證非零項
    assert simplify(gamma.data[0, 1, 1] - (-r)) == 0  # Gamma^r_tt
    assert simplify(gamma.data[1, 0, 1] - (1/r)) == 0 # Gamma^t_rt
    assert simplify(gamma.data[1, 1, 0] - (1/r)) == 0 # Gamma^t_tr
    
    # 驗證零項 (例如 Gamma^r_rr)
    assert gamma.data[0, 0, 0] == 0

def test_ricci_scalar_sphere(sphere_metric_setup):
    """
    測試球面的里奇純量 (Ricci Scalar)
    對於半徑 R 的球面，高斯曲率 K = 1/R^2
    2D 流形的 Ricci Scalar R = 2K
    對於單位球面 (R=1)，R 應該等於 2
    """
    metric, coords = sphere_metric_setup
    
    # 計算里奇純量
    R_scalar = metric.ricci_scalar()
    
    # 驗證 R = 2
    # 注意: 三角函數簡化有時需要 explicitly 呼叫 simplify
    assert simplify(R_scalar - 2) == 0

def test_geodesic_equations_polar(polar_metric_setup):
    """
    測試極座標下的測地線微分方程符號生成
    方程應為:
    1. r'' - r(theta')^2 = 0
    2. theta'' + (2/r) r' theta' = 0
    """
    metric, (r, theta) = polar_metric_setup
    tau = symbols('tau')
    
    eqs = metric.get_geodesic_equations(param_var=tau)
    
    # 定義預期的函數形式
    r_func = Function(r.name)(tau)
    theta_func = Function(theta.name)(tau)
    
    # 驗證第一個方程 (r 方向)
    # eqs[0] 結構是 Eq(Derivative(r, tau, tau), RHS)
    # 我們移項檢查: LHS - RHS == 0
    # 程式碼回傳的是 x'' = RHS，所以我們檢查 x'' - RHS
    
    # 預期: r'' - r (theta')^2 = 0 -> RHS = r (theta')^2
    # 注意程式碼回傳的 RHS 是 -Term，所以 x'' = -(-r theta'^2)
    
    lhs_0 = eqs[0].lhs
    rhs_0 = eqs[0].rhs
    expected_rhs_0 = r_func * (simplify(sp.diff(theta_func, tau))**2)
    
    assert simplify(lhs_0 - sp.diff(r_func, tau, tau)) == 0
    assert simplify(rhs_0 - expected_rhs_0) == 0

def test_arc_length_circle(polar_metric_setup):
    """測試弧長積分：計算圓周長"""
    metric, (r, theta) = polar_metric_setup
    t = symbols('t')
    
    # 參數式：半徑 r=5 的圓，theta=t
    # 路徑: r(t)=5, theta(t)=t
    path = [5, t]
    
    # 積分範圍 0 到 2pi
    length_integral = metric.arc_length(path, t, 0, 2*pi)
    
    # 執行積分
    length = length_integral.doit()
    
    # 2 * pi * R = 10 * pi
    assert simplify(length - 10*pi) == 0

# ==========================================
# 數值測試 (需要 numpy/scipy)
# ==========================================

def test_solve_geodesic_bvp_euclidean():
    """
    數值測試：在平面笛卡爾坐標系中，測地線應該是直線。
    """
    pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    
    x, y = symbols('x y')
    metric = TensorMetric([[1, 0], [0, 1]], [x, y])
    
    start = [0.0, 0.0]
    end = [1.0, 1.0]
    
    # 【修正】: num_points 改為 21 (奇數)，確保 index 10 正好落在 0.5 的位置
    # num_points=20 -> index 10 is ~0.526
    # num_points=21 -> index 10 is 0.500
    path = metric.solve_geodesic_bvp(start, end, num_points=21)
    
    # path shape: (2, 21)
    # 驗證路徑中間點 (index 10) 應該接近 (0.5, 0.5)
    mid_index = 10
    mid_point = path[:, mid_index]
    
    # 寬鬆容許誤差 (數值解)
    assert abs(mid_point[0] - 0.5) < 1e-2
    assert abs(mid_point[1] - 0.5) < 1e-2
    
    # 驗證是否在 y=x 線上
    assert np.allclose(path[0, :], path[1, :], atol=1e-2)
    
def test_curvature_flat_space_is_zero(polar_metric_setup):
    """
    物理驗證：極座標雖然看起來複雜，但它描述的是平直空間 (Flat Space)。
    其黎曼曲率張量所有分量必須為 0。
    """
    metric, _ = polar_metric_setup
    R_tensor = metric.riemann_tensor()
    
    # 檢查 rank
    assert R_tensor.index_config == [1, -1, -1, -1]
    
    # 檢查所有分量為 0
    # 展開 array 檢查
    flat_data = np.array(R_tensor.data).flatten()
    for val in flat_data:
        assert simplify(val) == 0