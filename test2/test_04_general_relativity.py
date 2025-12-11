import sympy as sp
from dgeom.sym import *

def test_schwarzschild_efe():
    print("==================================================")
    print("廣義相對論測試: 史瓦西度規 (Schwarzschild Metric)")
    print("驗證目標: 真空解的愛因斯坦張量 G_uv 是否為零")
    print("==================================================")

    # 1. 計算 Ricci Scalar
    print("2. 正在計算 Ricci Scalar (R)...")
    R = SCHWARZSCHILD_METRIC.ricci_scalar()
    print("   Ricci Scalar 結果 (預期為 0):")
    sp.pprint(R)
    
    # [Assert] 驗證純量曲率為 0
    assert sp.simplify(R) == 0, f"驗證失敗: Ricci Scalar 應為 0，但計算結果為 {R}"
    print("   -> [PASS] Ricci Scalar 驗證通過。")
    
    # 2. 計算愛因斯坦張量 G_uv
    print("\n3. 正在計算愛因斯坦張量 G_uv = R_uv - 1/2 R g_uv ...")
    G_tensor = SCHWARZSCHILD_METRIC.einstein_tensor()
    print("   愛因斯坦張量結果:")
    sp.pprint(G_tensor)

    # 3. 驗證重力場方程式 (真空: T_uv = 0)
    print("\n4. 驗證重力場方程式 (Vacuum EFE: G_uv = 0)...")
    efe_residual = SCHWARZSCHILD_METRIC.gravity_field_equations(T_uv=None)
    
    # 化簡結果
    simplified_residual = sp.simplify(efe_residual)
    
    if simplified_residual != sp.zeros(4, 4):
        print("殘餘項 (Error term):")
        sp.pprint(simplified_residual)

    # [Assert] 驗證愛因斯坦張量為零矩陣
    assert simplified_residual == sp.zeros(4, 4), "驗證失敗: 愛因斯坦張量 G_uv 應為全零矩陣 (真空解)。"
    print("\n[驗證成功] G_uv 是一個全零矩陣。")
    print("結論: 此度規滿足愛因斯坦真空場方程式。")


def test_derive_schwarzschild_radius():
    print("==================================================")
    print("廣義相對論測試: 導出史瓦西半徑 (Schwarzschild Radius)")
    print("方法: 尋找度規的時間分量 g_tt 為 0 的位置 (事件視界)")
    print("==================================================")

    metric = SCHWARZSCHILD_METRIC

    # 2. 提取時間分量 g_tt
    g_tt = metric.g[0, 0]
    print("1. 提取時間分量 g_tt:")
    sp.pprint(g_tt)
    
    # 3. 設定方程式 g_tt = 0 並求解 r
    print("\n2. 設定 g_tt = 0 (尋找事件視界)，並對 r 求解...")
    solutions = sp.solve(g_tt, r)
    print(f"   求解結果 (r 的解): {solutions}")

    # 4. 分析解
    print("\n3. 分析結果:")
    derived_Rs = None
    
    for sol in solutions:
        if sol != 0:
            derived_Rs = sol
            print("   找到非零解 (候選史瓦西半徑):")
            sp.pprint(derived_Rs)
            break
            
    # [Assert] 確保找到了解
    assert derived_Rs is not None, "驗證失敗: 未能找到非零的史瓦西半徑解。"

    # 5. 驗證是否符合 R_s = 2GM/c^2
    target_Rs = 2 * G * M / c**2
    
    print("-" * 40)
    # [Assert] 驗證導出的半徑公式正確
    assert sp.simplify(derived_Rs - target_Rs) == 0, \
        f"驗證失敗: 導出的半徑 {derived_Rs} 與預期 {target_Rs} 不符。"
    
    print("[驗證成功] 導出的解與標準史瓦西半徑公式 (2GM/c^2) 完全一致！")

    # --------------------------------------------------
    # 額外展示: 牛頓極限 (Weak Field Limit)
    # --------------------------------------------------
    print("\n==================================================")
    print("額外測試: 牛頓極限驗證 (Weak Field Limit)")
    print("目標: 證明 g_tt ~ -(c^2 + 2*Phi)，其中 Phi = -GM/r")
    print("==================================================")
    
    # 牛頓重力位勢
    Phi = -G * M / r
    newtonian_g_tt = c**2 + 2*Phi
    
    print("度規中的 g_tt:")
    sp.pprint(g_tt)
    print("牛頓極限預期值 (c^2 + 2*Phi):")
    sp.pprint(newtonian_g_tt)
    
    # [Assert] 驗證牛頓極限
    diff = sp.simplify(g_tt - newtonian_g_tt)
    assert diff == 0, "驗證失敗: 此度規未能在弱場下還原牛頓重力位勢。"
    
    print("\n[驗證成功] 此度規在弱場下精確還原了牛頓重力位勢。")


def test_flrw_metric_friedmann_equations():
    print("==================================================")
    print("廣義相對論測試: FLRW 度規 (宇宙學)")
    print("驗證目標: 檢查愛因斯坦張量形式是否符合弗里德曼方程式結構")
    print("==================================================")

    flrw_metric = FLRW_METRIC
    # --------------------------------------------------
    # 3. 計算愛因斯坦張量 G_uv
    # --------------------------------------------------
    print("正在計算愛因斯坦張量 (涉及對 a(t) 的微分，請稍候)...")
    G_tensor = flrw_metric.einstein_tensor()
    
    # --------------------------------------------------
    # 4. 驗證幾何特性 (Assertion)
    # --------------------------------------------------
    print("\n[驗證 1] 檢查非對角項 (Off-diagonal terms)...")
    # 在均勻各向同性宇宙中，G_uv 的非對角項必須為 0 (例如 G_tr, G_ttheta)
    # 這裡我們檢查 G_01 (t, r)
    g_tr = sp.simplify(G_tensor[0, 1])
    assert g_tr == 0, f"錯誤: G_tr 應為 0，但得到 {g_tr}"
    print("   -> [PASS] 非對角項 G_tr 為 0 (符合均勻性)。")

    print("\n[驗證 2] 檢查空間各向同性 (Isotropy)...")
    # G_theta_theta 與 G_phi_phi 應該有幾何關聯: G_33 = G_22 * sin^2(theta)
    g_22 = G_tensor[2, 2] # theta-theta
    g_33 = G_tensor[3, 3] # phi-phi
    
    difference = sp.simplify(g_33 - g_22 * sp.sin(theta)**2)
    assert difference == 0, "錯誤: G_phiphi 與 G_thetatheta 不符合球對稱關係。"
    print("   -> [PASS] G_phiphi = G_thetatheta * sin^2(theta) (符合各向同性)。")

    # --------------------------------------------------
    # 5. 展示弗里德曼方程式 (Friedmann Equations)
    # --------------------------------------------------
    print("\n[結果展示] 弗里德曼方程式的成分:")
    
    # G_tt 對應第一弗里德曼方程式 (描述能量密度 rho)
    # G_tt = 3 * (H^2 + k*c^2/a^2) * (相關係數)
    G_tt = sp.simplify(G_tensor[0, 0])
    
    print("\n1. 時間分量 G_tt (關聯到能量密度):")
    sp.pprint(G_tt)
    
    # 為了讓結果更容易閱讀，我們手動提取係數並驗證結構
    # 第一弗里德曼方程式通常包含: (dot(a)/a)^2 和 k/a^2
    print("   觀察: 應包含 a(t) 的一次微分平方 (adot^2) 與曲率 k")

    # G_rr 對應第二弗里德曼方程式 (描述壓力 p 與加速度)
    G_rr = sp.simplify(G_tensor[1, 1])
    
    print("\n2. 徑向分量 G_rr (關聯到壓力/加速度):")
    sp.pprint(G_rr)
    print("   觀察: 應包含 a(t) 的二次微分 (addot) 與一次微分平方")

    # --------------------------------------------------
    # 自動化結構檢查 (進階)
    # --------------------------------------------------
    # 檢查 G_tt 是否真的包含 dot(a)^2
    has_adot = G_tt.has(sp.Derivative(a, t))
    assert has_adot, "警告: G_tt 中未發現 a(t) 的時間導數，這不符合 FLRW 物理。"
    
    print("\n[總結] FLRW 度規測試完成，愛因斯坦張量結構正確。")

def test_mercury_precession_term():
    print("==================================================")
    print("廣義相對論測試: 水星近日點進動 (Mercury's Perihelion Precession)")
    print("驗證目標: 檢查有效位勢 (Effective Potential) 中是否存在 1/r^3 修正項")
    print("==================================================")
    
    M, G, c = sp.symbols('M G c', real=True, positive=True)
    metric = SCHWARZSCHILD_METRIC
    
    # 1. 設定軌道參數
    # 我們考慮位於赤道面上的軌道 (theta = pi/2)，簡化計算
    print("1. 設定赤道面軌道條件 (theta = pi/2)...")
    
    # 定義四速度分量 (u_t, u_r, u_theta, u_phi)
    # 使用符號變數代表對原時 tau 的微分: dt/dtau, dr/dtau, ...
    dt_dtau = sp.Symbol('udot_t')
    dr_dtau = sp.Symbol('udot_r')
    dphi_dtau = sp.Symbol('udot_phi')
    
    # 2. 利用守恆量 (Killing Vectors)
    # 由於度規不含 t 和 phi，存在兩個守恆量：
    # E (能量相關) 和 L (角動量相關)
    # g_tt * dt/dtau = E (常數)
    # g_phiphi * dphi/dtau = -L (常數)
    
    E, L = sp.symbols('E L', real=True) # 單位質量的能量與角動量常數
    
    # 在赤道面: theta = pi/2, sin(theta)=1
    g_tt = metric.g[0, 0]
    g_rr = metric.g[1, 1]
    g_phiphi = metric.g[3, 3].subs(theta, sp.pi/2) # -r^2
    
    print("   度規分量 (赤道面):")
    print(f"   g_tt = {g_tt}")
    print(f"   g_rr = {g_rr}")
    print(f"   g_pp = {g_phiphi}")
    
    # 3. 建立四速度歸一化方程式 (Normalization Condition)
    # 對於有質量粒子: g_uv u^u u^v = c^2 (使用 +--- 簽名慣例，且 ds^2 = c^2 dtau^2)
    # 注意: 這裡的 c^2 正負號取決於您度規定義的 ds^2 物理意義。
    # 您的度規 g_tt 是正的，代表 ds^2 > 0 為類時。
    # 方程式: g_tt(dt)^2 + g_rr(dr)^2 + g_pp(dphi)^2 = c^2
    
    print("\n2. 代入守恆量並列出徑向方程式...")
    
    # 將 dt 和 dphi 用 E, L 取代
    # dt/dtau = E / g_tt
    # dphi/dtau = -L / g_phiphi  (負號是慣例，平方後沒差)
    
    eqn = g_tt * (E / g_tt)**2 + g_rr * dr_dtau**2 + g_phiphi * (L / g_phiphi)**2
    
    # 這是總能量方程式，通常寫成 E_total = c^2
    total_energy_eqn = sp.simplify(eqn - c**2)
    
    # 4. 導出有效位勢 V_eff
    # 我們將方程式改寫為: (dr/dtau)^2 + V_eff(r) = 常數
    # 為了得到標準形式，我們需要隔離 dr_dtau^2
    
    # 解出 dr_dtau^2
    # 注意: g_rr 是負的 (-1/(1-Rs/r))，這會改變不等式方向，我們用 solve 直接處理等式
    sol_dr_sq = sp.solve(total_energy_eqn, dr_dtau**2)[0]
    
    # 這裡 sol_dr_sq 代表 (E^2 - V_eff_scaled) 的形式
    # 我們將其展開，尋找含有 r 的項
    expanded_eq = sp.expand(sol_dr_sq)
    
    print("\n3. 徑向動能 (dr/dtau)^2 的展開式:")
    sp.pprint(expanded_eq)
    
    # 5. 分析各項係數 (Coefficient Analysis)
    # 我們預期看到以下幾項 (忽略常數係數差異):
    # 1. 常數項 (與能量有關)
    # 2. 1/r 項 (牛頓重力位勢)
    # 3. 1/r^2 項 (離心力位勢)
    # 4. 1/r^3 項 (廣義相對論修正項 !!!) <-- 這是驗證關鍵
    
    coeff_r_inv3 = expanded_eq.coeff(1/r, 3) # 提取 (1/r)^3 的係數
    
    print("\n4. 檢查廣義相對論修正項 (1/r^3 的係數):")
    sp.pprint(coeff_r_inv3)
    
    # --------------------------------------------------
    # 驗證斷言 (Assertions)
    # --------------------------------------------------
    
    # 驗證 A: 修正項必須存在 (不為 0)
    assert coeff_r_inv3 != 0, "驗證失敗: 找不到 1/r^3 項，這看起來像牛頓力學而非廣義相對論。"
    
    # 驗證 B: 修正項的符號與結構
    # 對於史瓦西度規，該項應為 -R_s * L^2 (或類似結構，取決於 E/L 定義)
    # 從物理上講，這是一個吸引位勢 (負號)，導致軌道進動
    
    # 我們的 R_s = 2GM/c^2
    # 預期的結構包含 G, M, L, c
    has_G = coeff_r_inv3.has(G)
    has_M = coeff_r_inv3.has(M)
    has_L = coeff_r_inv3.has(L)
    
    assert has_G and has_M and has_L, "驗證失敗: 修正項缺少必要的物理常數 (G, M, L)。"
    
    # 驗證 C: 係數檢查
    # 物理分析:
    # 有效位勢 V_eff 中的修正項為 - (GM L^2 / c^2 r^3)，是負值 (吸引力)。
    # 但我們計算的是徑向動能 (dr/dtau)^2 = E_total - V_eff。
    # 因此，在 (dr/dtau)^2 的表達式中，該項係數應為 "正值" (- (- 負值))。
    
    # 由於符號較複雜，我們代入正數進行數值符號檢查
    check_sign = coeff_r_inv3.subs({G:1, M:1, L:1, c:1, R_s: 2})
    
    print(f"   數值代入檢查 (預期為正值): {check_sign}")

    # 修正: 這裡應該檢查 > 0
    assert check_sign > 0, f"驗證失敗: 在動能方程式中，1/r^3 項係數應為正 (代表位勢中的吸引力)，但得到 {check_sign}"

    print("-" * 40)
    print(f"[驗證成功] 發現 GR 修正項係數: {check_sign} (正值)")
    print("解釋: 動能項中的正係數對應於有效位勢中的負修正項 (額外吸引力)。")
    print("正是這個 1/r^3 項導致了橢圓軌道的近日點進動！")

if __name__ == "__main__":
    test_schwarzschild_efe()
    print("\n" + "#" * 50 + "\n")
    test_derive_schwarzschild_radius()
    test_flrw_metric_friedmann_equations()
    test_mercury_precession_term()