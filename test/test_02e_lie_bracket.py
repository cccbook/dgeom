import pytest
import sympy as sp
import numpy as np
from dgeom.sym import TangentVector, lie_bracket

# ===================================================================
# I. 基礎設定 (Fixtures)
# ===================================================================

@pytest.fixture
def coords():
    x, y, z = sp.symbols('x y z')
    return [x, y, z]

# ===================================================================
# II. 測試案例
# ===================================================================

def test_lie_bracket_zero(coords):
    """
    測試: 兩個相同向量場的李括號應為零。
    [V, V] = 0
    """
    x, y, z = coords
    
    # 向量場 V = x ∂x + y ∂y
    V = TangentVector([x, y, 0], coords, name="V")
    
    # 計算 [V, V]
    bracket_VV = lie_bracket(V, V)
    
    # 驗證: 結果應為 0 向量
    # GeometricTensor.data 是 NDimArray
    for val in np.array(bracket_VV.data).flatten():
        assert sp.simplify(val) == 0


def test_lie_bracket_non_zero(coords):
    """
    測試: 兩個非交換向量場的李括號應為非零。
    U = ∂x
    V = x ∂y - y ∂x
    
    理論計算:
    [∂x, x ∂y - y ∂x] 
    = ∂x(x ∂y) - ∂x(y ∂x) - (x ∂y(∂x) ...)
    = (∂x x) ∂y + x [∂x, ∂y] - ...
    = 1 * ∂y
    結果應為 ∂y (分量 [0, 1, 0])
    """
    x, y, z = coords
    
    U = TangentVector([1, 0, 0], coords, name="U")
    V = TangentVector([-y, x, 0], coords, name="V")

    # 計算 [U, V]
    bracket_UV = lie_bracket(U, V)

    # 預期結果: ∂y -> [0, 1, 0]
    expected_data = [0, 1, 0]
    
    for i, val in enumerate(bracket_UV.data):
        assert sp.simplify(val - expected_data[i]) == 0


def test_lie_bracket_antisymmetry(coords):
    """
    測試: 反對稱性 (Antisymmetry)
    [U, V] = -[V, U]
    """
    x, y, z = coords
    
    U = TangentVector([1, 0, 0], coords, name="U")
    V = TangentVector([-y, x, 0], coords, name="V")

    # 計算正反括號
    bracket_UV = lie_bracket(U, V)
    bracket_VU = lie_bracket(V, U)

    # 驗證 sum 為 0
    # Tensor 加法: bracket_UV.data + bracket_VU.data
    # 注意: SymPy NDimArray 支援直接加法
    sum_tensor = bracket_UV.data + bracket_VU.data
    
    for val in np.array(sum_tensor).flatten():
        assert sp.simplify(val) == 0


def test_lie_bracket_jacobian_identity(coords):
    """
    測試: 雅可比恆等式 (Jacobi Identity)
    [[U, V], W] + [[V, W], U] + [[W, U], V] = 0
    
    設定:
    U = ∂x
    V = ∂y
    W = x ∂z
    """
    x, y, z = coords
    
    U = TangentVector([1, 0, 0], coords, name="U")
    V = TangentVector([0, 1, 0], coords, name="V")
    W = TangentVector([0, 0, x], coords, name="W")

    # 1. 計算內層括號
    UV = lie_bracket(U, V) # 0
    VW = lie_bracket(V, W) # 0 (因為 W 只跟 x 有關，V 是 d/dy)
    WU = lie_bracket(W, U) # [x dz, dx] = -dx(x) dz = -dz
    
    # 2. 計算外層括號
    term1 = lie_bracket(UV, W) # [0, W] = 0
    term2 = lie_bracket(VW, U) # [0, U] = 0
    term3 = lie_bracket(WU, V) # [-dz, dy] = 0
    
    # 3. 總和
    total_data = term1.data + term2.data + term3.data
    
    for val in np.array(total_data).flatten():
        assert sp.simplify(val) == 0

def test_lie_bracket_complex_case(coords):
    """
    測試: 較複雜的函數組合，確保 Product Rule 處理正確
    U = y ∂x
    V = x ∂y
    
    [y ∂x, x ∂y] 
    = y [∂x, x ∂y] + (∂x y) ...
    = y ( (∂x x) ∂y ) - x ( (∂y y) ∂x )
    = y ∂y - x ∂x
    結果分量: [-x, y, 0]
    """
    x, y, z = coords
    
    U = TangentVector([y, 0, 0], coords)
    V = TangentVector([0, x, 0], coords)
    
    bracket = lie_bracket(U, V)
    
    # 預期 [-x, y, 0]
    expected = [-x, y, 0]
    
    for i, val in enumerate(bracket.data):
        assert sp.simplify(val - expected[i]) == 0

# ===================================================================
# 執行與除錯
# ===================================================================
if __name__ == "__main__":
    # 手動執行時的輸出
    print("--- Running Lie Bracket Tests ---")
    
    # 模擬 fixtures
    x, y, z = sp.symbols('x y z')
    c = [x, y, z]
    
    try:
        test_lie_bracket_zero(c)
        print("[PASS] Zero Bracket")
        
        test_lie_bracket_non_zero(c)
        print("[PASS] Non-Zero Bracket")
        
        test_lie_bracket_antisymmetry(c)
        print("[PASS] Antisymmetry")
        
        test_lie_bracket_jacobian_identity(c)
        print("[PASS] Jacobi Identity")
        
        test_lie_bracket_complex_case(c)
        print("[PASS] Complex Case")
        
    except AssertionError as e:
        print(f"[FAIL] {e}")
    except Exception as e:
        print(f"[ERROR] {e}")