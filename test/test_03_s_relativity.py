import pytest
import sympy as sp
import numpy as np
from dgeom.sym import minkowski_metric

# ===================================================================
# 幾何平坦性驗證
# ===================================================================
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

# ===================================================================
# 時空間距 ds 驗證
# ===================================================================
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

# ===================================================================
# 鐘慢效應 (Time Dilation) - 路徑積分法
# ===================================================================
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

# ===================================================================
# 尺縮效應 (Length Contraction) - 座標變換法
# ===================================================================
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

# ===================================================================
# 雙生子佯謬 (Twin Paradox) - 比較世界線長度
# ===================================================================
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