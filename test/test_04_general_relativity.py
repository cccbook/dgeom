import sympy as sp
from dgeom.sym import get_schwarzschild_metric, get_flrw_metric, get_minkowski_metric

# 史瓦西度規愛因斯坦場方程驗證
# AI 解說: https://gemini.google.com/share/7d284cd27d68
def test_schwarzschild_efe():
    print("==================================================")
    print("廣義相對論測試: 史瓦西度規 (Schwarzschild Metric)")
    print("驗證目標: 真空解的愛因斯坦張量 G_uv 是否為零")
    print("==================================================")

    # 1. 取得度規實例
    metric = get_schwarzschild_metric()
    
    # 【修正關鍵】從 metric 物件中取出符號，否則後面會報 NameError
    t, r, theta, phi = metric.coords

    # 2. 計算 Ricci Scalar
    print("2. 正在計算 Ricci Scalar (R)...")
    R = metric.ricci_scalar()
    print("   Ricci Scalar 結果 (預期為 0):")
    sp.pprint(R)
    
    # [Assert] 驗證純量曲率為 0
    assert sp.simplify(R) == 0, f"驗證失敗: Ricci Scalar 應為 0"
    print("   -> [PASS] Ricci Scalar 驗證通過。")
    
    # 3. 計算愛因斯坦張量 G_uv
    print("\n3. 正在計算愛因斯坦張量 G_uv...")
    G_tensor = metric.einstein_tensor()
    
    print("   愛因斯坦張量結果:")
    sp.pprint(G_tensor)

    # 4. 驗證重力場方程式
    print("\n4. 驗證重力場方程式 (Vacuum EFE: G_uv = 0)...")
    
    # 檢查是否為零矩陣
    # 注意: 因為 G_tensor 可能包含未完全化簡的項，這裡我們明確做一次 simplify
    simplified_G = sp.simplify(G_tensor)
    
    if simplified_G == sp.zeros(4, 4):
        print("\n[驗證成功] G_uv 是一個全零矩陣。")
    else:
        print("\n[驗證結果] G_uv 未完全化簡為零。")
        sp.pprint(simplified_G)
        assert False, "驗證失敗: 愛因斯坦張量應為零"

# 導出史瓦西半徑驗證
# AI 解說: https://gemini.google.com/share/05e2b99ac62e
def test_derive_schwarzschild_radius():
    print("==================================================")
    print("廣義相對論測試: 導出史瓦西半徑")
    print("==================================================")

    metric = get_schwarzschild_metric()
    
    # 【修正關鍵】解包符號: r 用於 solve, theta 用於定義結構
    t, r, theta, phi = metric.coords
    
    # 這裡的 M, G, c 是我們要用來驗證的外部常數，需重新定義
    M, G, c = sp.symbols('M G c', real=True, positive=True)

    # 2. 提取時間分量 g_tt
    g_tt = metric.g[0, 0]
    print("1. 提取時間分量 g_tt:")
    sp.pprint(g_tt)
    
    # 3. 設定方程式 g_tt = 0 並求解 r
    print("\n2. 設定 g_tt = 0，並對 r 求解...")
    # 這裡現在使用的是上一行解包出來的 r，不會報錯了
    solutions = sp.solve(g_tt, r)
    print(f"   求解結果: {solutions}")

    # 4. 分析解
    derived_Rs = None
    for sol in solutions:
        if sol != 0:
            derived_Rs = sol
            break
            
    assert derived_Rs is not None, "未找到非零解"

    target_Rs = 2 * G * M / c**2
    
    # 驗證
    if sp.simplify(derived_Rs - target_Rs) == 0:
        print(f"[驗證成功] 導出的解 {derived_Rs} 正確。")
    else:
        assert False, f"驗證失敗: 預期 {target_Rs}, 得到 {derived_Rs}"

# FLRW 度規驗證
# AI 解說: https://gemini.google.com/share/ebc6669a75a7
def test_flrw_metric_friedmann_equations():
    print("==================================================")
    print("廣義相對論測試: FLRW 度規")
    print("==================================================")
    
    flrw_metric = get_flrw_metric()
    
    # 【修正關鍵】解包符號: 用於後續 assert 檢查 sin(theta)
    t, r, theta, phi = flrw_metric.coords
    
    print("正在計算愛因斯坦張量...")
    G_tensor = flrw_metric.einstein_tensor()
    
    print("\n[驗證] 檢查空間各向同性 (Isotropy)...")
    # G_33 = G_22 * sin^2(theta)
    g_22 = G_tensor[2, 2] 
    g_33 = G_tensor[3, 3] 
    
    # 這裡現在使用的是解包出來的 theta
    difference = sp.simplify(g_33 - g_22 * sp.sin(theta)**2)
    assert difference == 0, "錯誤: G_phiphi 與 G_thetatheta 不符合球對稱關係。"
    print("   -> [PASS] 符合各向同性。")

    # 檢查 G_tt 結構
    G_tt = sp.simplify(G_tensor[0, 0])
    a = sp.Function('a')(t) # 重新定義一個相同的函數符號用於比對
    
    # 注意: 雖然符號名稱一樣，但確保我們檢查的是結構
    has_derivative = G_tt.has(sp.Derivative)
    assert has_derivative, "G_tt 中未發現時間導數項"
    
    print("\n[總結] FLRW 結構測試完成。")


# 水星進動項驗證
# AI 解說: https://gemini.google.com/share/0c982d642403
def test_mercury_precession_term():
    print("==================================================")
    print("廣義相對論測試: 水星進動項")
    print("==================================================")

    metric = get_schwarzschild_metric()
    
    # 【修正關鍵】解包符號: 需要 theta 代入數值
    t, r, theta, phi = metric.coords
    
    M, G, c = sp.symbols('M G c', real=True, positive=True)
    
    # 1. 設定軌道參數
    g_tt = metric.g[0, 0]
    g_rr = metric.g[1, 1]
    
    # 這裡使用解包出來的 theta
    g_phiphi = metric.g[3, 3].subs(theta, sp.pi/2) 
    
    E, L = sp.symbols('E L', real=True)
    dr_dtau = sp.Symbol('dr_dtau')
    
    # 建立方程式
    eqn = g_tt*(E/g_tt)**2 + g_rr*dr_dtau**2 + g_phiphi*(L/g_phiphi)**2 - c**2
    
    # 解出 (dr/dtau)^2
    sol = sp.solve(eqn, dr_dtau**2)[0]
    expanded = sp.expand(sol)
    
    # 提取 1/r^3 係數 (GR 修正項)
    coeff = expanded.coeff(1/r, 3)
    
    print("1/r^3 項係數:", coeff)
    
    # 驗證
    # 這裡手動定義 R_s 數值以便檢查符號
    # 注意 coeff 裡面包含的是 metric 內部的符號，但因為是純量係數，結構上是一樣的
    # 我們使用 .subs 代入數值最保險
    
    # 為了讓 .subs 成功，我們需要獲取 coeff 中實際使用的 G, M, c 符號
    # 這些符號來自 get_schwarzschild_metric 內部
    # 最簡單的方法是全部用字串比對，或者直接代入所有同名符號
    
    # 這裡我們利用 SymPy 的強大替換功能，它會匹配同名符號
    # 構造一個包含所有可能符號的字典
    check_sign = coeff.subs({
        sp.Symbol('G', real=True, positive=True): 1,
        sp.Symbol('M', real=True, positive=True): 1,
        sp.Symbol('c', real=True, positive=True): 1,
        sp.Symbol('L', real=True): 1
    })
    
    # 動能項中的係數應為正 (對應位勢中的負吸引力)
    assert check_sign > 0, f"驗證失敗: 係數應為正，得到 {check_sign}"

    print("[驗證成功] 發現水星進動修正項。")

# GPS 衛星時間膨脹效應驗證
# AI 解說: https://gemini.google.com/share/99e4559d5aa3
def test_gps_satellite_time_dilation():
    r"""
    ### 🧪 驗證 GPS 衛星的時間膨脹效應 (GPS Time Dilation)
    
    **測試目標**: 
    驗證人造衛星(GPS)上的原子鐘相對於地球表面時鐘的走時差異。
    
    **物理來源**:
    1. 狹義相對論 (SR): 衛星高速運動 -> 時間變慢 (約 -7.2 us/day)
    2. 廣義相對論 (GR): 衛星重力較弱 -> 時間變快 (約 +45.9 us/day)
    3. 總和效應: 衛星時鐘每天快約 38.7 us
    
    **數學模型**:
    利用史瓦西度規計算原時 (Proper Time) 的流逝速率 $d\tau/dt$。
    $$ d\tau = \sqrt{g_{00} + g_{11}v_r^2 + \dots} \, dt $$
    """
    print("\n==================================================")
    print("廣義相對論測試: GPS 衛星時間膨脹 (SR + GR)")
    print("==================================================")

    # 1. 初始化度規與符號
    metric = get_schwarzschild_metric()
    t, r, theta, phi = metric.coords
    
    # 定義物理常數符號 (用於代入數值)
    sym_G = sp.Symbol('G', real=True, positive=True)
    sym_M = sp.Symbol('M', real=True, positive=True)
    sym_c = sp.Symbol('c', real=True, positive=True)
    
    # 2. 定義真實物理數值 (GPS 參數)
    # 參考資料: GPS 軌道半徑約 26,560 km (地表高度 20,200 km)
    # 地球半徑 R_E ~ 6,371 km
    val_G = 6.67430e-11  # m^3 kg^-1 s^-2
    val_M = 5.9722e24    # kg (地球質量)
    val_c = 2.99792458e8 # m/s (光速)
    val_R_E = 6.371e6    # m (地球半徑)
    val_R_sat = 2.656e7  # m (衛星軌道半徑 = R_E + 20200km)
    val_v_sat = 3.874e3  # m/s (衛星軌道速度)
    
    seconds_in_day = 86400.0
    
    # 建立數值代入字典
    # 注意: get_schwarzschild_metric 內部使用的符號可能需要通過 .atoms() 或字串匹配來對應
    # 這裡我們利用 metric.g 裡的符號直接建立字典
    # 為了確保符號對應正確，我們從 metric 表達式中提取符號
    # 但為求簡便，這裡直接構造表達式時使用已知符號
    
    # 3. 建立原時流逝率公式 (Rate of Proper Time: dtau / dt)
    # 根據度規: c^2 dtau^2 = g_tt dt^2 + g_rr dr^2 + g_th th dtheta^2 + g_ph ph dphi^2
    # 我們比較單位座標時間 dt 內，原時 dtau 經過了多少
    # dtau/dt = (1/c) * sqrt( g_tt + g_rr(dr/dt)^2 + ... )
    
    # --- 情境 A: 地球表面的時鐘 (Earth Clock) ---
    # 條件: r = R_E, 速度 v=0 (忽略地球自轉，視為靜止參考系)
    # dr/dt = 0, dtheta/dt = 0, dphi/dt = 0
    g_tt_earth = metric.g[0, 0].subs(r, val_R_E)
    
    # dtau_earth / dt
    rate_earth_expr = sp.sqrt(g_tt_earth) / sym_c
    
    # --- 情境 B: GPS 衛星時鐘 (Satellite Clock) ---
    # 條件: r = R_sat, 具有切線速度 v_sat
    # 近似: v^2 = - (g_phiphi * (dphi/dt)^2 + ...) 
    # 在史瓦西度規中，空間部分是負的，切線速度 v 對應的項是 -v^2
    # dtau_sat / dt = (1/c) * sqrt( g_tt(R_sat) - v_sat^2 )
    
    g_tt_sat = metric.g[0, 0].subs(r, val_R_sat)
    
    # 注意: 度規中的 g_tt 包含 c^2，所以是 c^2 * (1 - Rs/r)
    # 速度項 v^2 也是物理速度平方
    rate_sat_expr = sp.sqrt(g_tt_sat - val_v_sat**2) / sym_c

    # 4. 進行數值計算
    # 建立常數替換表 (尋找 metric 中對應 G, M, c 的符號物件)
    # 技巧：透過 atoms 過濾出符號
    params = {
        s: val for s, val in zip([sym_G, sym_M, sym_c], [val_G, val_M, val_c])
    }
    # 實際上 metric 內的符號是獨立的，我們需要讓表達式裡的符號被替換
    # 使用字串名稱匹配最穩健
    symbols_in_metric = metric.g.free_symbols
    subs_dict = {}
    for s in symbols_in_metric:
        if s.name == 'G': subs_dict[s] = val_G
        elif s.name == 'M': subs_dict[s] = val_M
        elif s.name == 'c': subs_dict[s] = val_c
    
    # 計算速率
    rate_earth_val = rate_earth_expr.subs(subs_dict).evalf()
    rate_sat_val = rate_sat_expr.subs(subs_dict).evalf()
    
    print(f"1. 地球時鐘流逝率 (dtau/dt): {rate_earth_val:.16f}")
    print(f"2. 衛星時鐘流逝率 (dtau/dt): {rate_sat_val:.16f}")
    
    # 5. 計算一天的累積誤差 (微秒)
    # 差異 = (衛星速率 - 地球速率) * 一天秒數
    # 如果衛星速率 > 地球速率，代表衛星過得比較快，差異為正
    diff_per_day_seconds = (rate_sat_val - rate_earth_val) * seconds_in_day
    diff_per_day_us = diff_per_day_seconds * 1e6 # 換算成微秒
    
    print(f"\n3. 每天的時間差異: {diff_per_day_us:.4f} 微秒 (us)")
    
    # --------------------------------------------------
    # 驗證分析
    # --------------------------------------------------
    
    # 預期結果: 約 +38.7 us
    # 容許誤差: +/- 1.0 us (因為軌道參數近似值可能略有不同)
    expected_diff = 38.7
    tolerance = 1.0 
    
    print(f"   預期值 (來自文章): +{expected_diff} us")
    
    assert abs(diff_per_day_us - expected_diff) < tolerance, \
        f"驗證失敗: 計算出的時間差 {diff_per_day_us} 與預期值 {expected_diff} 差異過大"

    print("-> [PASS] 總體時間膨脹效應驗證成功 (符合 GPS 系統修正值)。")

    # --------------------------------------------------
    # 加分題: 分離 SR 與 GR 效應 (驗證文章的細項)
    # --------------------------------------------------
    print("\n--- 詳細效應分解驗證 ---")
    
    # GR 效應 (重力紅移): 假設衛星靜止 (v=0)，只比較高度差異
    rate_sat_gr_only = (sp.sqrt(g_tt_sat) / sym_c).subs(subs_dict).evalf()
    diff_gr = (rate_sat_gr_only - rate_earth_val) * seconds_in_day * 1e6
    print(f"   GR 效應 (重力): {diff_gr:.2f} us (預期約 +45.9)")
    
    # SR 效應 (速度): 假設在平坦時空 (無重力) 或相同高度，只比較速度
    # 近似計算: 衛星比靜止慢的量 = (sqrt(1 - v^2/c^2) - 1) * T
    sr_factor = sp.sqrt(1 - (val_v_sat**2 / val_c**2))
    diff_sr = (sr_factor - 1) * seconds_in_day * 1e6
    print(f"   SR 效應 (速度): {diff_sr:.2f} us (預期約 -7.2)")
    
    # 驗證 SR 與 GR 的方向性
    assert diff_gr > 40, "GR 效應應顯著為正 (快)"
    assert diff_sr < -5, "SR 效應應顯著為負 (慢)"
    
    print("-> [PASS] 效應分離驗證成功 (GR變快, SR變慢)。")

if __name__ == "__main__":
    test_schwarzschild_efe()
    test_derive_schwarzschild_radius()
    test_flrw_metric_friedmann_equations()
    test_mercury_precession_term()
    test_gps_satellite_time_dilation()
