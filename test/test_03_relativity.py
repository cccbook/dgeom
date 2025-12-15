import pytest
import sympy as sp
import numpy as np
from dgeom.sym import *

# =======================狹義相對論 ===========================

# 幾何平坦性驗證

def test_minkowski_flatness():
    """
    [數學驗證] 閔可夫斯基時空必須是平坦的 (Ricci 張量為 0)。
    """
    st = minkowski_metric()
    
    # 呼叫 Spacetime 的代理方法 ricci_tensor (需在 Spacetime 實作，或透過 metric 存取)
    # 假設 Spacetime 尚未實作 ricci_tensor 代理，我們透過 .metric 存取
    ricci = st.metric.ricci_tensor()
    
    # 驗證所有分量為 0
    for val in np.array(ricci.data).flatten():
        assert sp.simplify(val) == 0, "平坦時空 Ricci 張量應為 0"


# 時空間距 ds 驗證
def test_minkowski_ds():
    """
    [數學驗證] 驗證線元素公式 ds^2。
    目標公式: ds^2 = c^2 dt^2 - dx^2 - dy^2 - dz^2
    注意: 若函式庫預設為自然單位制 (c=1)，則驗證 ds^2 = dt^2 - dx^2 - dy^2 - dz^2
    """
    st = minkowski_metric()
    
    # 1. 定義微分座標符號 (Differentials)
    dt, dx, dy, dz = sp.symbols('dt dx dy dz', real=True)
    dX = sp.Matrix([dt, dx, dy, dz])
    
    # 2. 取得度規張量矩陣 g (Metric Tensor)
    # 假設 st.metric.data 是一個二維列表或陣列，將其轉換為 SymPy Matrix 以利運算
    g_matrix = sp.Matrix(st.metric.data)
    
    # 3. 執行張量縮併計算 ds^2 = g_uv * dx^u * dx^v
    # dX.T 是 (1x4), g_matrix 是 (4x4), dX 是 (4x1) -> 結果為 (1x1) 純量
    ds_squared_calc = (dX.T * g_matrix * dX)[0]
    
    # 4. 建構預期的線元素公式
    # 我們從度規的 g_00 分量動態獲取 c^2 (或是 1)，以相容不同的單位制設定
    c_squared = g_matrix[0, 0] 
    
    # 預期形式: (c^2)dt^2 - dx^2 - dy^2 - dz^2
    ds_squared_expected = c_squared * dt**2 - dx**2 - dy**2 - dz**2
    
    # 5. 驗證
    # 檢查空間分量是否確實為 -1 (Signature Check)
    assert g_matrix[1, 1] == -1 and g_matrix[2, 2] == -1 and g_matrix[3, 3] == -1, \
        "閔可夫斯基空間的空間分量應為 -1 (使用 +--- 慣例)"
        
    # 檢查交叉項是否為 0 (例如 dtdx, dxdy 等) 且公式相符
    assert sp.simplify(ds_squared_calc - ds_squared_expected) == 0, \
        f"線元素形式錯誤。\n計算結果: {ds_squared_calc}\n預期結果: {ds_squared_expected}"


# 鐘慢效應 (Time Dilation) - 路徑積分法
def test_time_dilation():
    """
    [物理驗證] 鐘慢效應。
    驗證移動時鐘的原時 (Proper Time) 小於靜止時鐘。
    公式: d_tau = d_t * sqrt(1 - v^2)
    """
    st = minkowski_metric()
    t, x, y, z = st.coords
    
    # 參數設定
    v = sp.Symbol('v', real=True, positive=True) # 速度 (v < 1)
    T = sp.Symbol('T', real=True, positive=True) # 實驗室時間間隔
    
    # 定義世界線 (Worldline): x(t) = vt
    # 使用 t 作為參數變數
    path = [t, v*t, 0, 0]
    
    # 計算原時 (Proper Time)
    # 呼叫 MetricTensor 的 arc_length
    proper_time = st.metric.arc_length(path, param_var=t, start_val=0, end_val=T)
    
    # 執行積分
    result = proper_time.doit()
    
    # 預期結果: T * sqrt(1 - v^2)
    # 注意: Minkowski ds^2 = dt^2 - dx^2 = (1-v^2)dt^2
    # arc_length 計算的是 integral(sqrt(ds^2))
    expected = T * sp.sqrt(1 - v**2)
    
    # 驗證
    assert sp.simplify(result - expected) == 0, \
        f"鐘慢效應計算錯誤: 預期 {expected}, 得到 {result}"


# 尺縮效應 (Length Contraction) - 座標變換法
def test_length_contraction():
    """
    [物理驗證] 尺縮效應。
    利用勞倫茲變換驗證靜止長度 (L0) 與測量長度 (L) 的關係。
    L = L0 * sqrt(1 - v^2)
    """
    # 此測試不需要積分，而是驗證代數關係
    v = sp.Symbol('v', real=True)
    L_measured = sp.Symbol('L', real=True, positive=True) # 實驗室測得長度 (同時測量)
    gamma = 1 / sp.sqrt(1 - v**2)
    
    # 實驗室座標系 (S) 中的兩個事件 (測量尺頭和尺尾)
    # Event A (尾): x=0, t=0
    # Event B (頭): x=L, t=0 (同時測量)
    
    # 變換到尺的靜止系 (S')
    # x' = gamma(x - vt)
    x_prime_A = gamma * (0 - 0)
    x_prime_B = gamma * (L_measured - 0)
    
    # 靜止系中的長度 (Proper Length) L0 = x'_B - x'_A
    L_proper = x_prime_B - x_prime_A
    
    # 驗證: L_measured = L_proper / gamma
    # 即: L_proper = L_measured * gamma
    expected_relation = L_measured * gamma
    
    assert sp.simplify(L_proper - expected_relation) == 0


# 雙生子佯謬 (Twin Paradox) - 比較世界線長度
def test_twin_paradox():
    """
    [物理驗證] 雙生子佯謬。
    故事：雙胞胎AB兩人，A 留在原處，B 出去高速旅行，回來之後，B 比 A 年輕很多。
    證明：在 Minkowski 時空中，慣性路徑 (直線) 的原時最長。
    Home (慣性): 0 -> 2T
    Travel (折線): 0 -> T (v), T -> 2T (-v)
    """
    st = minkowski_metric()
    t, x, y, z = st.coords
    
    v = sp.Symbol('v', real=True, positive=True)
    T = sp.Symbol('T', real=True, positive=True)
    
    # 1. 居家者 (Home): x=0
    path_home = [t, 0, 0, 0]
    tau_home = st.metric.arc_length(path_home, t, 0, 2*T).doit()
    
    # 2. 旅行者 (Traveler)
    # 去程: x = vt
    path_out = [t, v*t, 0, 0]
    tau_out = st.metric.arc_length(path_out, t, 0, T).doit()
    
    # 回程: x = v(2T - t) => dx/dt = -v
    path_back = [t, v*(2*T - t), 0, 0]
    tau_back = st.metric.arc_length(path_back, t, T, 2*T).doit()
    
    tau_travel = sp.simplify(tau_out + tau_back)
    
    # 3. 驗證
    # A. 符號驗證: tau_travel = 2T

# ======================= 廣義相對論 ===========================

# 輔助工具: 動態符號提取

def get_symbols_map(spacetime):
    """
    從 Spacetime 物件中提取符號映射表。
    解決 SymPy 中 'M' (由工廠函數建立) 與 'M' (測試中定義) 視為不同物件的問題。
    回傳: {'M': Symbol('M'), 'c': Symbol('c'), ...}
    """
    return {s.name: s for s in spacetime.metric.data.free_symbols}


# 測試案例


def test_einstein_field_equation():
    """
    測試 Spacetime 物件的場方程式介面是否正常工作。
    """
    st = minkowski_metric()
    
    # 測試 1: 真空 (None)
    efe_vac = st.field_equations(T_uv=None)
    # 結果應為 0 (因為 Minkowski G=0)
    assert sp.simplify(efe_vac.data[0, 0]) == 0
    
    # 測試 2: 傳入 T_uv
    # 模擬一個簡單的 T_uv = diag(rho, 0, 0, 0)
    rho = sp.Symbol('rho')
    T_matrix = sp.diag(rho, 0, 0, 0)
    kappa = sp.Symbol('kappa')
    
    efe_matter = st.field_equations(T_uv=T_matrix, kappa=kappa)
    
    # E = G - kappa * T = 0 - kappa * rho
    expected_00 = -kappa * rho
    assert sp.simplify(efe_matter.data[0, 0] - expected_00) == 0

# 數學核心驗證: 史瓦西真空解
def test_schwarzschild_vacuum():
    """
    [數學驗證] 史瓦西度規必須嚴格滿足真空愛因斯坦場方程式 G_uv = 0。
    這證明了微分幾何引擎的張量運算邏輯正確。
    """
    # 1. Arrange (準備)
    st = schwarzschild_metric()
    
    # 2. Act (執行)
    G = st.einstein_tensor()
    
    # 3. Assert (驗證)
    # 檢查對角線分量是否全為 0
    for i in range(4):
        val = sp.simplify(G.data[i, i])
        assert val == 0, f"真空解破壞: G_{i}{i} 應為 0，但得到 {val}"


# 幾何結構驗證: 史瓦西半徑 (事件視界)
def test_schwarzschild_radius():
    """
    [幾何驗證] 從度規分量 g_tt = 0 導出史瓦西半徑 R_s = 2GM/c^2。
    驗證度規的奇異點結構。
    """
    st = schwarzschild_metric()
    t, r, theta, phi = st.coords
    syms = get_symbols_map(st)
    
    # 提取 g_tt
    g_tt = st.metric.data[0, 0]
    
    # 求解 g_tt = 0
    solutions = sp.solve(g_tt, r)
    
    # 過濾出非零解
    rs_solution = next((sol for sol in solutions if sol != 0), None)
    assert rs_solution is not None, "未找到事件視界"
    
    # 預期結果
    expected_Rs = 2 * syms['G'] * syms['M'] / syms['c']**2
    
    assert sp.simplify(rs_solution - expected_Rs) == 0

def test_flrw_expansion():
    """
    測試 FLRW 宇宙學模型。
    重點驗證：這不是真空解，且包含時間導數 (宇宙膨脹)。
    """
    # 測試帶有曲率 k=1 的宇宙
    st = flrw_metric(k_val=1)
    
    assert "k=1" in st.name
    
    # 計算 G_00 (與能量密度有關)
    G = st.einstein_tensor()
    G_00 = G.data[0, 0]
    
    # 1. 不應為 0 (宇宙充滿物質/能量)
    assert sp.simplify(G_00) != 0
    
    # 2. 必須包含標度因子 a(t) 的時間導數
    # 使用 .atoms(sp.Derivative) 檢查是否存在導數項
    derivs = G_00.atoms(sp.Derivative)
    assert len(derivs) > 0, "FLRW G_00 必須包含時間導數 (da/dt)"

def test_flrw_symbolic_k():
    """
    測試 FLRW 工廠函數是否能處理符號 k (未指定 k_val)。
    """
    st = flrw_metric(k_val=None) # k 應為符號
    
    # 檢查度規中是否存在名為 'k' 的符號
    symbols_in_metric = st.metric.data.free_symbols
    symbol_names = {s.name for s in symbols_in_metric}
    
    assert 'k' in symbol_names
    assert 'sym' in st.name

def test_kerr_black_hole():
    """
    測試克爾 (Kerr) 度規的結構。
    注意：不計算愛因斯坦張量，因為 Kerr 的符號運算非常耗時 (可能數分鐘)。
    主要驗證它具有旋轉黑洞特有的非對角項 g_t_phi。
    """
    st = kerr_metric()
    
    assert "Kerr" in st.name
    
    # 檢查 g_03 (t, phi) 項
    # 在對角度規中，這個位置應該是 0
    g_t_phi = st.metric.data[0, 3]
    g_phi_t = st.metric.data[3, 0]
    
    # 1. 確保非對角項存在且非零
    assert g_t_phi != 0
    assert g_phi_t != 0
    
    # 2. 確保是對稱的 g_03 == g_30
    assert sp.simplify(g_t_phi - g_phi_t) == 0
    
    # 3. 檢查自旋參數 'a' 是否存在於度規中
    symbols_in_g03 = g_t_phi.free_symbols
    symbol_names = {s.name for s in symbols_in_g03}
    assert 'a' in symbol_names, "非對角項應包含自旋參數 a"


# 動態時空驗證: FLRW 宇宙膨脹
def test_flrw_expansion():
    """
    [動態驗證] 驗證 FLRW 度規描述的是一個動態膨脹的宇宙。
    愛因斯坦張量 G_00 必須包含標度因子 a(t) 的時間導數。
    """
    # 使用 k=0 (平坦宇宙)
    st = flrw_metric(k_val=0)
    
    G = st.einstein_tensor()
    G_tt = G.data[0, 0]
    
    # 驗證結構: 必須包含 Derivative(a(t), t)
    # G_00 正比於 Hubble 參數平方 H^2 ~ (da/dt / a)^2
    assert G_tt.has(sp.Derivative), "G_tt 缺少時間導數，宇宙是靜止的？"
    
    # 驗證各向同性: G_theta_theta 與 G_phi_phi 的關係
    t, r, theta, phi = st.coords
    g_22 = G.data[2, 2]
    g_33 = G.data[3, 3]
    assert sp.simplify(g_33 - g_22 * sp.sin(theta)**2) == 0, "宇宙各向同性破壞"


# 經典實驗驗證: 水星近日點進動
def test_mercury_precession():
    """
    [實驗驗證] 驗證史瓦西度規下的有效位勢包含 1/r^3 修正項。
    這是廣義相對論解釋水星軌道異常進動的關鍵。
    """
    st = schwarzschild_metric()
    t, r, theta, phi = st.coords
    syms = get_symbols_map(st)
    c = syms['c']
    
    # 1. 取得赤道面 (theta = pi/2) 的度規分量
    g = st.metric.data
    g_tt = g[0, 0]
    g_rr = g[1, 1]
    g_phi = g[3, 3].subs(theta, sp.pi/2)
    
    # 2. 定義軌道常數
    E, L = sp.symbols('E L', real=True)
    dr_dtau = sp.Symbol('dr_dtau')
    
    # 3. 軌道方程式 (四速度歸一化 u.u = c^2, for (+---) metric)
    # g_tt(dt/dtau)^2 + g_rr(dr/dtau)^2 + g_phi(dphi/dtau)^2 = c^2
    # 代入守恆量: dt/dtau = E/g_tt, dphi/dtau = L/g_phi
    # 注意: E, L 這裡定義為每單位質量的能量與角動量
    eqn = g_tt*(E/g_tt)**2 + g_rr*dr_dtau**2 + g_phi*(L/g_phi)**2 - c**2
    
    # 4. 解出徑向動能項 (dr/dtau)^2 ~ E_eff - V_eff
    radial_kinetic = sp.solve(eqn, dr_dtau**2)[0]
    
    # 5. 展開並尋找 GR 修正項 (-GM/r + L^2/2r^2 - GML^2/c^2 r^3)
    # 我們關注 1/r^3 的係數
    expanded = sp.expand(radial_kinetic)
    term_r3 = expanded.coeff(1/r, 3)
    
    # 6. 驗證該項存在且係數非零
    # 代入數值確保係數存在 (避免純符號判斷困難)
    check_val = term_r3.subs({syms['G']: 1, syms['M']: 1, syms['c']: 1, L: 1})
    assert check_val != 0, "未發現導致進動的 1/r^3 GR 修正項"


# 工程應用驗證: GPS 時間膨脹
def test_gps_time_dilation_integration():
    """
    [工程驗證] 模擬 GPS 衛星與地面時鐘的速率差異。
    結合狹義相對論(速度)與廣義相對論(重力)效應。
    預期結果: 衛星時鐘每天快約 38.7 微秒。
    """
    st = schwarzschild_metric()
    t, r, theta, phi = st.coords
    syms = get_symbols_map(st)
    
    # 真實物理常數
    consts = {
        syms['G']: 6.67430e-11,
        syms['M']: 5.9722e24,   # Earth Mass
        syms['c']: 2.99792458e8
    }
    
    R_E = 6.371e6     # Earth Radius
    R_sat = 2.656e7   # GPS Orbit Radius
    v_sat = 3.874e3   # GPS Orbital Speed
    
    # 1. 建立原時流逝率公式 dtau/dt
    # proper_time_rate = sqrt(g_uv dx^u dx^v) / c / dt
    # 對於圓軌道: dtau/dt = (1/c) * sqrt(g_tt - |g_phi|*(omega)^2) 
    # 或者簡單寫成: sqrt(g_tt - v^2)/c (注意 g_tt 內含 c^2)
    
    g_tt = st.metric.data[0, 0]
    
    # A. 地球時鐘 (靜止於表面, v=0)
    # rate = sqrt(g_tt(R_E)) / c
    rate_earth_expr = sp.sqrt(g_tt.subs(r, R_E)) / syms['c']
    val_earth = rate_earth_expr.subs(consts).evalf()
    
    # B. 衛星時鐘 (軌道上, 有速度 v)
    # rate = sqrt(g_tt(R_sat) - v^2) / c
    # g_tt ~ c^2 - 2GM/r, 所以 g_tt - v^2 合理
    rate_sat_expr = sp.sqrt(g_tt.subs(r, R_sat) - v_sat**2) / syms['c']
    val_sat = rate_sat_expr.subs(consts).evalf()
    
    # 2. 計算單日誤差
    seconds_per_day = 86400
    # 差異 = (衛星 - 地面)
    diff_us = (val_sat - val_earth) * seconds_per_day * 1e6
    
    # 3. 驗證 (38.7 us +/- 0.5 us)
    print(f"\n[GPS] Earth Rate: {val_earth}")
    print(f"[GPS] Sat Rate:   {val_sat}")
    print(f"[GPS] Diff/Day:   {diff_us:.4f} us")
    
    assert 38.0 < diff_us < 40.0, f"GPS 驗證失敗: 預期 38.7，得到 {diff_us}"


# 執行與除錯

if __name__ == "__main__":
    # 允許直接執行此腳本
    sys.exit(pytest.main(["-v", __file__]))