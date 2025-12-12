from dgeom.sym import line_integral
import sympy as sp

# --------------------------------------------------
# V. 線積分測試 (Test Cases)
# --------------------------------------------------

def test_line_integral():
    print("--- 線積分測試開始 ---")
    
    # 宣告符號
    x, y, z = sp.symbols('x y z')
    t = sp.symbols('t')
    
    # 1. 測試案例：保守場 (期望結果：與路徑無關，只取決於端點)
    print("\n[測試 1] 保守場 F = ∇f, f = x*y*z")
    
    # 向量場 F = [yz, xz, xy]
    F1 = sp.Matrix([y*z, x*z, x*y])
    
    # 路徑 C1: 圓弧 r(t) = [cos(t), sin(t), t], t 介於 0 到 π/2
    path_r1 = sp.Matrix([sp.cos(t), sp.sin(t), t])
    ta1, tb1 = 0, sp.pi/2
    
    # 預期結果: f(r(π/2)) - f(r(0))
    # r(π/2) = (0, 1, π/2) -> f(0, 1, π/2) = 0
    # r(0)   = (1, 0, 0)   -> f(1, 0, 0) = 0
    expected1 = 0
    
    result1 = line_integral(F1, path_r1, t, ta1, tb1)
    print(f"向量場 F1: {F1}")
    print(f"路徑 r1(t): {path_r1.T}")
    print(f"計算結果: {result1}")
    print(f"預期結果: {expected1} {'(PASS)' if result1 == expected1 else '(FAIL)'}")
    
    # 2. 測試案例：非保守場 (路徑依賴)
    print("\n[測試 2] 非保守場 F = [-y, x, 0]")
    
    # 向量場 F = [-y, x, 0]
    F2 = sp.Matrix([-y, x, 0])
    
    # 路徑 C2: 單位圓 r(t) = [cos(t), sin(t), 0], t 介於 0 到 2π (封閉迴路)
    path_r2 = sp.Matrix([sp.cos(t), sp.sin(t), 0])
    ta2, tb2 = 0, 2*sp.pi
    
    # 預期結果: ∫_0^{2π} (sin²(t) + cos²(t)) dt = ∫_0^{2π} 1 dt = 2π
    expected2 = 2 * sp.pi
    
    result2 = line_integral(F2, path_r2, t, ta2, tb2)
    print(f"向量場 F2: {F2}")
    print(f"路徑 r2(t): {path_r2.T}")
    print(f"計算結果: {result2}")
    print(f"預期結果: {expected2} {'(PASS)' if result2 == expected2 else '(FAIL)'}")
    
    print("\n--- 線積分測試結束 ---")

# 執行測試
if __name__ == '__main__':
    # 確保 vcalculus.py 的基礎符號已定義
    x, y, z = sp.symbols('x y z') 
    
    test_line_integral()