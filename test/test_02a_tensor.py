import sympy as sp
from sympy import MutableDenseNDimArray, Matrix, diff
from dgeom.sym import GeometricTensor
import pytest

# ==========================================
# Pytest 測試案例
# ==========================================

@pytest.fixture
def polar_setup():
    """
    Fixtures: 設定笛卡爾與極座標的符號及變換規則
    """
    x, y = sp.symbols('x y')
    r, theta = sp.symbols('r theta', real=True, positive=True)
    
    rules = {
        x: r * sp.cos(theta),
        y: r * sp.sin(theta)
    }
    
    return (x, y), (r, theta), rules

def test_vector_transformation(polar_setup):
    """
    測試逆變向量 (Contravariant Vector) 的變換: V^i
    笛卡爾: [x, y] -> 極座標: [r, 0]
    """
    (x, y), (r, theta), rules = polar_setup
    
    # 定義 V = x \hat{x} + y \hat{y}
    V_cart = GeometricTensor([x, y], [x, y], [1])
    
    # 執行變換
    V_polar = V_cart.transform([r, theta], rules)
    
    # Assertions
    assert V_polar.coords == [r, theta]
    assert V_polar.index_config == [1]
    
    # 驗證分量數值: [r, 0]
    # 使用 sp.simplify(a - b) == 0 來比較符號表達式
    assert sp.simplify(V_polar.data[0] - r) == 0
    assert sp.simplify(V_polar.data[1]) == 0

def test_metric_transformation(polar_setup):
    """
    測試協變張量 (Metric Tensor) 的變換: g_ij
    笛卡爾: I -> 極座標: diag(1, r^2)
    """
    (x, y), (r, theta), rules = polar_setup
    
    # 定義 Euclidean Metric
    g_cart = GeometricTensor([[1, 0], [0, 1]], [x, y], [-1, -1])
    
    # 執行變換
    g_polar = g_cart.transform([r, theta], rules)
    
    # Assertions
    assert g_polar.index_config == [-1, -1]
    
    # 驗證分量: [[1, 0], [0, r^2]]
    assert sp.simplify(g_polar.data[0, 0] - 1) == 0
    assert sp.simplify(g_polar.data[0, 1]) == 0
    assert sp.simplify(g_polar.data[1, 0]) == 0
    assert sp.simplify(g_polar.data[1, 1] - r**2) == 0

def test_scalar_invariant(polar_setup):
    """
    測試純量不變量 (Invariant): V^2 = g_ij V^i V^j
    驗證不管在哪個座標系計算，長度平方都應該是 r^2 (即 x^2 + y^2)
    """
    (x, y), (r, theta), rules = polar_setup
    
    # 準備極座標下的張量
    V_cart = GeometricTensor([x, y], [x, y], [1])
    g_cart = GeometricTensor([[1, 0], [0, 1]], [x, y], [-1, -1])
    
    V_polar = V_cart.transform([r, theta], rules)
    g_polar = g_cart.transform([r, theta], rules)
    
    # 計算 g_ij (x) V^k -> Rank 3 Tensor
    T_mixed = g_polar.tensor_product(V_polar)
    
    # 下標化 (Lowering Index): g_ij V^j -> V_i
    # 縮併 index 1 (g的第二個下標) 和 index 2 (V的上標)
    V_covector = T_mixed.contract(1, 2)
    
    # 驗證中間產物 V_i = [r, 0]
    assert sp.simplify(V_covector.data[0] - r) == 0
    assert sp.simplify(V_covector.data[1]) == 0
    
    # 最後縮併 V_i V^i -> Scalar
    scalar_tensor = V_covector.tensor_product(V_polar).contract(0, 1)
    
    # 取出純量數值 (注意使用 [()] 存取 Rank 0 array)
    scalar_value = scalar_tensor.data[()]
    
    # Assert: 結果應為 r^2
    assert sp.simplify(scalar_value - r**2) == 0

def test_mixed_tensor_transformation():
    """
    測試混合張量 T^i_j 的變換
    這裡測試 Identity 變換 T^i_j = delta^i_j
    """
    x, y = sp.symbols('x y')
    u, v = sp.symbols('u v')
    # 簡單線性變換: x = u + v, y = u - v
    rules = {x: u + v, y: u - v}
    
    # Kronecker delta
    delta_data = [[1, 0], [0, 1]]
    T_cart = GeometricTensor(delta_data, [x, y], [1, -1]) # Up, Down
    
    T_new = T_cart.transform([u, v], rules)
    
    # 混合張量如果是 Identity (Kronecker delta)，在任何座標系下都應該保持不變
    # 但這只對 Tensor 定義成立。如果是 Matrix 則不一定。
    # 數學上 delta^i_j 是不變張量。
    
    # 驗證 T'^i_j 依然是 identity
    assert sp.simplify(T_new.data[0, 0] - 1) == 0
    assert sp.simplify(T_new.data[1, 1] - 1) == 0
    assert sp.simplify(T_new.data[0, 1]) == 0
    assert sp.simplify(T_new.data[1, 0]) == 0

def test_error_handling():
    """
    測試錯誤捕捉機制
    """
    x, y = sp.symbols('x y')
    
    # 測試1: 縮併相同類型的指標 (應報錯)
    T = GeometricTensor([[1, 0], [0, 1]], [x, y], [1, 1]) # 兩個上標
    with pytest.raises(ValueError, match="必須縮併一個協變指標和一個逆變指標"):
        T.contract(0, 1)

    # 測試2: 不同座標系運算 (應報錯)
    z = sp.symbols('z')
    T1 = GeometricTensor([1, 0], [x, y], [1])
    T2 = GeometricTensor([1, 0], [x, z], [1])
    with pytest.raises(ValueError, match="必須在相同座標系下"):
        T1.tensor_product(T2)

