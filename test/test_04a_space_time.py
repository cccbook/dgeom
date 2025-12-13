import pytest
import sympy as sp
import numpy as np
from sympy import symbols, sin, cos, simplify, Function, diag
from dgeom.sym import TensorMetric, Spacetime

# ===================================================================
# Fixtures: 建立標準時空模型
# ===================================================================

@pytest.fixture
def minkowski_spacetime():
    """建立閔可夫斯基時空 (SR, Flat)"""
    t, x, y, z = symbols('t x y z', real=True)
    coords = [t, x, y, z]
    # diag(1, -1, -1, -1)
    g_data = diag(1, -1, -1, -1)
    
    metric = TensorMetric(g_data, coords)
    return Spacetime(metric, name="Minkowski")

@pytest.fixture
def schwarzschild_spacetime():
    """建立史瓦西時空 (Vacuum, Curved)"""
    t, r, theta, phi = symbols('t r theta phi', real=True)
    G, M, c = symbols('G M c', real=True, positive=True)
    
    R_s = 2 * G * M / c**2
    f_r = 1 - R_s / r
    
    g_data = diag(
        c**2 * f_r,
        -1 / f_r,
        -r**2,
        -r**2 * sin(theta)**2
    )
    coords = [t, r, theta, phi]
    metric = TensorMetric(g_data, coords)
    return Spacetime(metric, name="Schwarzschild")

@pytest.fixture
def flrw_spacetime():
    """建立 FLRW 時空 (Matter, Dynamic)"""
    t, r, theta, phi = symbols('t r theta phi', real=True)
    c = symbols('c', real=True, positive=True)
    a = Function('a')(t) # Scale factor
    k = symbols('k', real=True)
    
    D_r = 1 / (1 - k * r**2)
    
    g_data = diag(
        c**2,
        -a**2 * D_r,
        -a**2 * r**2,
        -a**2 * r**2 * sin(theta)**2
    )
    coords = [t, r, theta, phi]
    metric = TensorMetric(g_data, coords)
    return Spacetime(metric, name="FLRW")

# ===================================================================
# 測試案例
# ===================================================================

def test_initialization(minkowski_spacetime):
    """
    測試 Spacetime 物件的初始化與基本屬性
    """
    st = minkowski_spacetime
    
    assert st.name == "Minkowski"
    assert st.dim == 4
    # 確保 metric 屬性是 TensorMetric
    assert isinstance(st.metric, TensorMetric)
    # 確保 coords 正確傳遞
    assert str(st.coords[0]) == 't'

def test_type_error_on_init():
    """
    測試錯誤處理：Spacetime 必須接收 TensorMetric
    """
    x, y = symbols('x y')
    invalid_metric = [[1, 0], [0, 1]] # 這只是 list，不是 TensorMetric
    
    with pytest.raises(TypeError, match="必須基於一個 TensorMetric"):
        Spacetime(invalid_metric)

def test_delegation(minkowski_spacetime):
    """
    測試代理方法 (Delegation)：
    Spacetime 應該能直接呼叫 metric 的幾何方法 (如 christoffel_symbols)
    """
    st = minkowski_spacetime
    
    # 呼叫 st.christoffel_symbols() 應該委派給 st.metric.christoffel_symbols()
    gamma = st.christoffel_symbols()
    
    # 檢查回傳類型 (應該是 GeometricTensor)
    assert hasattr(gamma, 'index_config')
    assert gamma.index_config == [1, -1, -1]
    
    # 對於 Minkowski，Gamma 全為 0
    # flatten() 用於遍歷 NDimArray
    for val in np.array(gamma.data).flatten():
        assert simplify(val) == 0

def test_einstein_tensor_vacuum(schwarzschild_spacetime):
    """
    測試真空解 (Schwarzschild): G_uv 必須全為 0
    這驗證了物理層 (G = Ric - 0.5 R g) 與數學層的計算整合正確。
    """
    st = schwarzschild_spacetime
    
    G = st.einstein_tensor()
    
    # 檢查組態 (Rank 2 Covariant)
    assert G.index_config == [-1, -1]
    
    # 驗證 G_uv = 0 (真空)
    # 由於符號運算較慢，這裡只抽查對角線分量
    dim = st.dim
    for i in range(dim):
        # 使用 simplify 處理複雜的三角函數消去
        assert simplify(G.data[i, i]) == 0

def test_einstein_tensor_dynamic(flrw_spacetime):
    """
    測試動態解 (FLRW): G_uv 不為 0，且包含 a(t) 的導數
    """
    st = flrw_spacetime
    
    G = st.einstein_tensor()
    
    # 檢查 G_00 (時間分量)
    G_tt = G.data[0, 0]
    
    # 1. 數值不應為 0
    assert simplify(G_tt) != 0
    
    # 2. 應該包含 Derivative (因為 a 是 t 的函數)
    # G_00 ~ 3 (da/dt)^2 / a^2 + 3kc^2/a^2 ...
    assert G_tt.has(sp.Derivative)

def test_field_equations_input_handling(minkowski_spacetime):
    """
    測試場方程式方法的輸入處理
    1. T_uv = None (真空)
    2. T_uv = Matrix/List
    """
    st = minkowski_spacetime
    
    # 1. 真空測試: E = G
    # Minkowski 的 G=0
    efe_vacuum = st.field_equations(T_uv=None)
    for val in np.array(efe_vacuum.data).flatten():
        assert val == 0
        
    # 2. 物質測試: E = G - kappa * T
    # 構造一個 T_uv = diag(1, 0, 0, 0) (靜止塵埃)
    T_data = diag(1, 0, 0, 0)
    kappa = symbols('kappa')
    
    efe_matter = st.field_equations(T_uv=T_data, kappa=kappa)
    
    # G=0, 所以 E = -kappa * T
    # E_00 = -kappa * 1
    assert simplify(efe_matter.data[0, 0] - (-kappa)) == 0
    # E_11 = 0
    assert efe_matter.data[1, 1] == 0

def test_caching_mechanism(schwarzschild_spacetime):
    """
    測試快取機制: 第二次呼叫 einstein_tensor 應該回傳相同物件
    """
    st = schwarzschild_spacetime
    
    G1 = st.einstein_tensor()
    G2 = st.einstein_tensor()
    
    # 檢查物件識別碼 (Identity)
    assert G1 is G2