import sympy as sp
from dgeom.sym import get_schwarzschild_metric, get_flrw_metric, get_minkowski_metric

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

if __name__ == "__main__":
    test_schwarzschild_efe()
    test_derive_schwarzschild_radius()
    test_flrw_metric_friedmann_equations()
    test_mercury_precession_term()