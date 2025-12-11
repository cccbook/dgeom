from dgeom.sym import Metric, EUCLIDEAN_METRIC
import sympy as sp

def test_polar_coordinates():
    """
    測試: 在 2D 極坐標系下，計算克里斯多福符號、黎曼張量、里奇張量和純量曲率。
    度規: ds^2 = dr^2 + r^2 dθ^2
    """
    # 定義坐標變數
    r, theta = sp.symbols('r theta') 

    # 定義 2D 極坐標的度規矩陣
    g_polar = sp.Matrix([
        [1, 0],
        [0, r**2]
    ])

    # 創建 Metric 物件
    polar_metric = Metric(g_polar, [r, theta])

    # 計算克里斯多福符號
    gamma = polar_metric.christoffel_symbols()
    
    # 驗證非零的克里斯多福符號分量
    assert gamma[0, 1, 1] == -r  # Gamma^r_{theta, theta}
    assert gamma[1, 0, 1] == 1/r  # Gamma^theta_{r, theta}

    # 計算黎曼曲率張量
    riemann = polar_metric.riemann_tensor()
    
    # 驗證黎曼張量分量 (2D 黎曼曲率只有一個獨立分量，且在平坦空間應為零)
    assert riemann[0, 1, 0, 1] == 0  # R^r_{theta, r, theta}

    # 計算里奇張量
    ricci = polar_metric.ricci_tensor()
    
    # 驗證里奇張量為零矩陣 (平坦空間)
    assert ricci == sp.Matrix([[0, 0], [0, 0]])

    # 計算純量曲率
    scalar = polar_metric.ricci_scalar()
    
    # 驗證純量曲率為零
    assert scalar == 0

def test_polar_example():
    # --------------------------------------------------
    # 範例使用 (Example Usage)
    # --------------------------------------------------

    # 1. 定義坐標變數
    t, r, theta = sp.symbols('t r theta') 

    # 2. 定義 2D 極坐標 (Polar Coordinates) 的度規矩陣 (Metric Matrix)
    # ds^2 = dr^2 + r^2 d\theta^2
    g_polar = sp.Matrix([
        [1, 0],
        [0, r**2]
    ])

    # 3. 創建 Metric 物件
    polar_metric = Metric(g_polar, [r, theta])

    print("### 2D 極坐標度規計算範例 ###")

    # 4. 計算克里斯多福符號 (Christoffel Symbols)
    # Christoffel 符號 Gamma^k_{ij} [k, i, j]
    gamma = polar_metric.christoffel_symbols()
    print("\n- Christoffel 符號 (Gamma^k_{ij}):")
    # 只有 Gamma^r_{theta, theta} 和 Gamma^theta_{r, theta} 是非零的
    # Gamma^r_{theta, theta} = -r
    gamma_r_theta_theta = gamma[0, 1, 1]
    print(rf"Gamma^r_{{theta, theta}}: {gamma_r_theta_theta}") 
    assert gamma_r_theta_theta == -r, "Gamma^r_{theta, theta} 計算錯誤"
    
    # Gamma^theta_{r, theta} = 1/r
    gamma_theta_r_theta = gamma[1, 0, 1]
    print(rf"Gamma^theta_{{r, theta}}: {gamma_theta_r_theta}") 
    assert gamma_theta_r_theta == 1/r, "Gamma^theta_{r, theta} 計算錯誤"

    # 5. 計算黎曼曲率張量 (Riemann Tensor)
    riemann = polar_metric.riemann_tensor()
    print("\n- 黎曼曲率張量 R^k_{lij}:")
    # 預期為 0 (2D 歐氏空間為零)
    riemann_r_theta_r_theta = riemann[0, 1, 0, 1]
    print(rf"R^r_{{theta, r, theta}}: {riemann_r_theta_r_theta}") 
    assert riemann_r_theta_r_theta == 0, "黎曼曲率張量分量 R^r_{theta, r, theta} 不為零"

    # 6. 計算里奇張量 (Ricci Tensor)
    ricci = polar_metric.ricci_tensor()
    # 預期為零矩陣 (平坦空間)
    expected_ricci = sp.Matrix([[0, 0], [0, 0]])
    print("\n- 里奇張量 (Ricci Tensor):")
    print(ricci) 
    assert ricci == expected_ricci, "里奇張量不為零矩陣"

    # 7. 計算純量曲率 (Scalar Curvature)
    scalar = polar_metric.ricci_scalar()
    print("\n- 純量曲率 (Scalar Curvature):")
    print(scalar) # 預期為 0
    assert scalar == 0, "純量曲率不為零"

    # 8. 計算曲線長度 (Arc Length)
    print("\n- 曲線弧長 (Arc Length) 範例:")
    # 考慮圓 r(t) = R, theta(t) = t, t in [0, 2*pi]
    R = sp.symbols('R', positive=True)
    path_r_theta = [R, t]
    length_integral = polar_metric.arc_length(path_r_theta, t, 0, 2*sp.pi)
    print(rf"圓的弧長積分表達式: {length_integral}")
    # 結果應為 2*pi*R
    result_length = length_integral.doit()
    print(rf"積分結果: {result_length}") 
    assert sp.simplify(result_length) == 2 * sp.pi * R, "圓弧長計算錯誤"

def test_special_relativity_metric():
    # --------------------------------------------------
    # 範例使用：狹義相對論 (Special Relativity) 測試
    # --------------------------------------------------
    print("\n" + "="*50)
    print("### 狹義相對論：閔可夫斯基空間測試案例 (時間項為 -1) ###")
    print("="*50)

    # 1. 定義坐標變數和常數
    t, x, y, z, tau_start, tau_end = sp.symbols('t x y z tau_start tau_end', real=True) 
    c = sp.symbols('c', positive=True)
    # 引入 x0 符號來代表 ct，作為微分坐標，解決 SymPy 複合微分問題
    x0 = sp.symbols('x_0') 

    # 坐標列表：我們使用獨立符號 [x^0, x^1, x^2, x^3] = [x0, x, y, z]
    coords_minkowski = [x0, x, y, z] # <--- 修正點 1：使用獨立符號

    # 2. 定義閔可夫斯基度規矩陣 (Metric Matrix)，時間項為負
    # eta_mu_nu = diag(-1, 1, 1, 1)
    g_minkowski = sp.Matrix([
        [-1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    # 3. 創建 Metric 物件
    minkowski_metric = Metric(g_minkowski, coords_minkowski) # <--- 修正點 2：使用新的坐標列表

    print("\n--- 幾何量計算 (平坦時空) ---")
    # ... (幾何量計算保持不變，應全部為 0) ...
    # 4. 計算克里斯多福符號 (應為零)
    gamma_m = minkowski_metric.christoffel_symbols()
    print(rf"克里斯多福符號的一個分量 Gamma^x_{{x0, x0}} (應為 0): {gamma_m[1, 0, 0]}")
    assert gamma_m[1, 0, 0] == 0, "閔可夫斯基空間的克里斯多福符號應為零"

    # 5. 計算黎曼曲率張量 (應為零)
    riemann_m = minkowski_metric.riemann_tensor()
    print(rf"黎曼曲率張量的一個分量 R^x_{{x0, x, x0}} (應為 0): {riemann_m[1, 0, 1, 0]}")
    assert riemann_m[1, 0, 1, 0] == 0, "閔可夫斯基空間的黎曼張量應為零"

    # 6. 計算里奇張量 (應為零)
    ricci_m = minkowski_metric.ricci_tensor()
    print(rf"里奇張量 (應為零矩陣):\n{ricci_m}")
    assert ricci_m == sp.zeros(4, 4), "閔可夫斯基空間的里奇張量應為零矩陣"

    # 7. 計算純量曲率 (應為零)
    scalar_m = minkowski_metric.ricci_scalar()
    print(rf"純量曲率 (應為 0): {scalar_m}")
    assert scalar_m == 0, "閔可夫斯基空間的純量曲率應為零"

    # --- 固有時間 / 距離 計算 ---
    print("\n--- 固有時間 / 距離計算 (0.5 倍光速) ---")

    # 假設運動路徑：沿 x 軸以恆定速度 v = 0.5c 運動
    v = sp.Rational(1, 2) * c 
    # 坐標路徑：[x^0(t), x^1(t), x^2(t), x^3(t)] = [c*t, v*t, 0, 0]
    path_param_minkowski = [c * t, v * t, 0, 0] # <--- 路徑參數化保持不變
    param_var_m = t
    t_start = 0
    t_end = tau_end 

    # 距離積分 (Distance Integral) = \int_{0}^{T} \sqrt{g_{ij} \frac{dx^i}{dt} \frac{dx^j}{dt}} dt
    # $g_{ij} \frac{dx^i}{dt} \frac{dx^j}{dt} = (-1)(c)^2 + (1)(v)^2 = v^2 - c^2$
    distance_integral = minkowski_metric.arc_length(path_param_minkowski, param_var_m, t_start, t_end)
    # 修正：使用 str() 轉換 SymPy 結果
    print(rf"\n距離積分 ($\int ds$): {str(distance_integral.doit())}") 
    assert sp.simplify(distance_integral.doit() - sp.sqrt(v**2 - c**2) * t_end) == 0, "距離積分計算錯誤"

    # 重新計算固有時間 (Proper Time) \tau = \int \sqrt{-\frac{1}{c^2} ds^2}
    proper_time_integrand = sp.sqrt(1 - (v/c)**2)
    proper_time_integral = sp.Integral(proper_time_integrand, (param_var_m, t_start, t_end))
    proper_time = proper_time_integral.doit()

    print(rf"\n運動路徑的固有時間 (Proper Time $\tau$):")
    # 修正：使用 str() 轉換 SymPy 結果，並將 LaTeX 中的 { 改為 {{
    print(rf"積分表達式 $\int \sqrt{{1 - v^2/c^2}} dt$: {str(proper_time_integral)}")
    # 修正：使用 str() 轉換 SymPy 結果
    print(rf"代入 v = 0.5c 後的結果: {str(proper_time.subs(v, sp.Rational(1, 2) * c))}")
    assert sp.simplify(proper_time - (sp.sqrt(3)/2) * t_end) == 0, "固有時間計算錯誤"

    # 驗證時間膨脹
    time_dilation_factor = proper_time.subs(v, sp.Rational(1, 2) * c) / t_end
    # 修正：使用 str() 轉換 SymPy 結果
    print(rf"時間膨脹因子 $\Delta\tau / \Delta t$: {str(time_dilation_factor)}")
    # 修正：將 LaTeX 中的 { 改為 {{
    print(rf"驗證 $\sqrt{{1 - 1/4}} = \sqrt{{3}}/2 \approx 0.866$: {time_dilation_factor.evalf()}")
    assert sp.simplify(time_dilation_factor - sp.sqrt(3)/2) == 0, "時間膨脹計算錯誤"