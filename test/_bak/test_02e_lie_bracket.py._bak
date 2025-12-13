import sympy as sp
from dgeom.sym.dgeometry import TangentVector, lie_bracket

# --------------------------------------------------
# I. 基礎設定
# --------------------------------------------------

# 座標變數
x, y, z = sp.symbols('x y z')
coords = [x, y, z]

# --------------------------------------------------
# II. 測試案例
# --------------------------------------------------

def test_lie_bracket_zero():
    """
    測試: 兩個平行或常數向量場的李括號應為零。
    [V, V] = 0
    """
    # 向量場 V = (x, y, 0)
    V_comp = [x, y, 0]
    V = TangentVector(V_comp, coords, name="V")
    
    # 應證 [V, V] = 0
    bracket_VV = lie_bracket(V, V)
    
    # 預期分量都應為 0
    expected_comp = sp.Matrix([0, 0, 0])
    
    assert bracket_VV.components == expected_comp
    print("Test Passed: [V, V] = 0")


def test_lie_bracket_non_zero():
    """
    測試: 兩個非交換向量場的李括號應為非零。
    U = d/dx
    V = x * d/dy - y * d/dx
    [U, V] = [d/dx, x * d/dy - y * d/dx]
           = [d/dx, x * d/dy] - [d/dx, y * d/dx]
           = x * [d/dx, d/dy] + (d/dx(x)) * d/dy - y * [d/dx, d/dx] - (d/dx(y)) * d/dx
           = 0 + 1 * d/dy - 0 - 0 
           = d/dy
    """
    # 向量場 U = d/dx (分量: [1, 0, 0])
    U_comp = [1, 0, 0]
    U = TangentVector(U_comp, coords, name="U")

    # 向量場 V = x * d/dy - y * d/dx (分量: [-y, x, 0])
    V_comp = [-y, x, 0]
    V = TangentVector(V_comp, coords, name="V")

    # 計算 [U, V]
    bracket_UV = lie_bracket(U, V)

    # 預期結果: d/dy (分量: [0, 1, 0])
    expected_comp = sp.Matrix([0, 1, 0])
    
    assert bracket_UV.components == expected_comp
    print("Test Passed: [U, V] = d/dy")


def test_lie_bracket_antisymmetry():
    """
    測試: 反對稱性 (Antisymmetry)
    [U, V] = -[V, U]
    使用 test_lie_bracket_non_zero 中的 U 和 V
    """
    # 向量場 U = d/dx (分量: [1, 0, 0])
    U_comp = [1, 0, 0]
    U = TangentVector(U_comp, coords, name="U")

    # 向量場 V = x * d/dy - y * d/dx (分量: [-y, x, 0])
    V_comp = [-y, x, 0]
    V = TangentVector(V_comp, coords, name="V")

    # 計算 [U, V]
    bracket_UV = lie_bracket(U, V)
    
    # 計算 [V, U]
    bracket_VU = lie_bracket(V, U)

    # 應證 [U, V] = -[V, U]
    expected_VU_comp = -bracket_UV.components
    
    assert bracket_VU.components == expected_VU_comp
    print("Test Passed: Antisymmetry [U, V] = -[V, U]")


def test_lie_bracket_jacobian_identity():
    """
    測試: 雅可比恆等式 (Jacobi Identity) - 簡化版
    [[U, V], W] + [[V, W], U] + [[W, U], V] = 0
    
    U = d/dx = [1, 0, 0]
    V = d/dy = [0, 1, 0]
    W = x * d/dz = [0, 0, x]
    
    1. [U, V] = [d/dx, d/dy] = 0
    2. [V, W] = [d/dy, x * d/dz] = d/dy(x) * d/dz - x * [d/dy, d/dz] = 0 - 0 = 0
    3. [W, U] = [x * d/dz, d/dx] = x * [d/dz, d/dx] + d/dx(x) * d/dz = 0 + 1 * d/dz = d/dz
       (分量: [0, 0, 1])
    
     LHS = [[0], W] + [[0], U] + [d/dz, V]
         = 0 + 0 + [d/dz, d/dy] = -[d/dy, d/dz] = 0
    """
    U = TangentVector([1, 0, 0], coords, name="U")
    V = TangentVector([0, 1, 0], coords, name="V")
    W = TangentVector([0, 0, x], coords, name="W")

    # 1. [U, V]
    UV = lie_bracket(U, V)
    
    # 2. [V, W]
    VW = lie_bracket(V, W)
    
    # 3. [W, U]
    WU = lie_bracket(W, U)
    
    # 計算三項
    term1 = lie_bracket(UV, W)
    term2 = lie_bracket(VW, U)
    term3 = lie_bracket(WU, V)
    
    # 總和
    total_components = term1.components + term2.components + term3.components
    
    expected_comp = sp.Matrix([0, 0, 0])
    
    assert total_components == expected_comp
    print("Test Passed: Jacobi Identity")


# 執行所有測試
if __name__ == "__main__":
    print("--- 執行李括號 (Lie Bracket) 測試 ---")
    test_lie_bracket_zero()
    test_lie_bracket_non_zero()
    test_lie_bracket_antisymmetry()
    test_lie_bracket_jacobian_identity()
    print("--- 所有測試完成 ---")