import pytest
import sympy as sp
import numpy as np
from dgeom.sym import minkowski_metric

# ===================================================================
# 測試 1: 幾何平坦性驗證
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
# 測試 2: 鐘慢效應 (Time Dilation) - 路徑積分法
# ===================================================================
def test_time_dilation_integration():
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
    # 呼叫 TensorMetric 的 arc_length
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
# 測試 3: 尺縮效應 (Length Contraction) - 座標變換法
# ===================================================================
def test_length_contraction_lorentz():
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
# 測試 4: 雙生子佯謬 (Twin Paradox) - 比較世界線長度
# ===================================================================
def test_twin_paradox_path_comparison():
    """
    [物理驗證] 雙生子佯謬。
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