from dgeom.sym.vcalculus import gradient, divergence, curl, line_integral
import sympy as sp

# --------------------------------------------------
# III. 測試 (Test Cases)
# --------------------------------------------------

def test_vector_calculus():
    print("--- 向量微積分測試開始 (直角坐標系) ---")
    
    x, y, z, t = sp.symbols('x y z t')
    
    # 純量場 f = x*y^2 + z
    f = x * y**2 + z
    
    # 向量場 F = [x^2, -2yz, xy]
    F = sp.Matrix([x**2, -2*y*z, x*y])
    
    
    ## 1. 梯度測試
    print("\n[測試 1] 梯度 ∇f")
    # 期望結果: [y^2, 2xy, 1]
    grad_result = gradient(f)
    expected_grad = sp.Matrix([y**2, 2*x*y, 1])
    print(f"計算結果: {grad_result.T}")
    print(f"預期結果: {expected_grad.T} {'(PASS)' if grad_result == expected_grad else '(FAIL)'}")
    
    ## 2. 散度測試
    print("\n[測試 2] 散度 ∇ · F")
    # 期望結果: 2x - 2z
    div_result = divergence(F)
    expected_div = 2*x - 2*z
    print(f"計算結果: {div_result}")
    print(f"預期結果: {expected_div} {'(PASS)' if div_result == expected_div else '(FAIL)'}")

    ## 3. 旋度測試
    print("\n[測試 3] 旋度 ∇ × F")
    # 期望結果: [x + 2y, -y, 0]
    curl_result = curl(F)
    expected_curl = sp.Matrix([x + 2*y, -y, 0])
    print(f"計算結果: {curl_result.T}")
    print(f"預期結果: {expected_curl.T} {'(PASS)' if curl_result == expected_curl else '(FAIL)'}")
    
    ## 4. 線積分測試 (使用 F = [x^2, -2yz, xy])
    print("\n[測試 4] 線積分 ∫_C F · dr")
    
    # 路徑 C: r(t) = [t, t^2, 1] 從 t=0 到 t=1
    path_r4 = sp.Matrix([t, t**2, 1])
    ta4, tb4 = 0, 1
    
    # 預期結果: -2/3
    expected_integral = sp.Rational(-2, 3)
    
    integral_result = line_integral(F, path_r4, t, ta4, tb4)
    print(f"向量場 F: {F.T}")
    print(f"路徑 r(t): {path_r4.T}")
    print(f"計算結果: {integral_result}")
    print(f"預期結果: {expected_integral} {'(PASS)' if integral_result == expected_integral else '(FAIL)'}")
    
    print("\n--- 向量微積分測試結束 ---")

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
    # 確保基礎符號已定義
    x, y, z = sp.symbols('x y z') 
    
    test_vector_calculus()
    test_line_integral()
