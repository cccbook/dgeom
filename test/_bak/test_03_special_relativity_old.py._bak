import sympy as sp
import pytest
from dgeom.sym import get_minkowski_metric

# --------------------------------------------------
# 閔可夫斯基度規的平坦性驗證
# AI 解說: https://gemini.google.com/share/df7e8b241da6
# --------------------------------------------------
def test_relativity_minkowski_flatness():
    r"""
    ### 🧪 驗證 relativity.py：閔可夫斯基度規的平坦性
    數學公式: $R_{\mu \nu} = 0$
    """
    # 1. 計算 Ricci 張量
    metric = get_minkowski_metric()
    # 由於 Minkowski 是平坦時空，Ricci 張量應全為 0
    R_mn = metric.ricci_tensor() 
    
    # 2. 驗證
    assert sp.simplify(R_mn) == sp.zeros(4, 4), \
        r"閔可夫斯基度規的 Ricci 張量 $R_{\mu \nu}$ 應為零"

# --------------------------------------------------
# 鐘慢效應 (Time Dilation) 的幾何驗證
# AI 解說: https://gemini.google.com/share/0af6296a6790
# --------------------------------------------------
def test_time_dilation():
    r"""
    ### 🧪 驗證狹義相對論：鐘慢效應 (Time Dilation)
    
    **物理原理**:
    移動的時鐘走得比較慢。
    若實驗室座標時經過 $\Delta t$，則移動速度為 $v$ 的時鐘，其原時 (Proper Time) $\Delta \tau$ 應為：
    $$ \Delta \tau = \Delta t \sqrt{1 - v^2/c^2} = \frac{\Delta t}{\gamma} $$
    
    此測試設定 $c=1$。
    """
    # 1. 初始化度規
    metric = get_minkowski_metric()
    t, x, y, z = metric.coords
    
    # 2. 定義參數
    v = sp.Symbol('v', real=True, positive=True) # 速度
    T = sp.Symbol('T', real=True, positive=True) # 實驗室經過的時間
    # 假設 v < 1 (小於光速 c=1)
    
    # 3. 定義一條移動中的時鐘的路徑 (Worldline)
    # 參數變數使用 t
    # 路徑: x(t) = v*t, y=0, z=0
    path = [t, v*t, 0, 0]
    
    # 4. 計算路徑的「弧長」 (即 Proper Time 原時)
    # arc_length 會計算積分: integral(sqrt(g_uv dx^u dx^v))
    # 在 Minkowski (+---) 下，ds^2 = dt^2 - dx^2 = dt^2 - v^2 dt^2 = (1-v^2)dt^2
    # 注意：Metric.arc_length 預設開根號。
    # 由於我們的簽名是 (+, -, -, -)，類時區間 ds^2 > 0，直接開根號沒問題。
    proper_time = metric.arc_length(path, t, 0, T)
    
    # 5. 理論預期值: T * sqrt(1 - v^2)
    expected_proper_time = T * sp.sqrt(1 - v**2)
    
    print(f"\n[鐘慢測試] 計算出的原時: {proper_time}")
    print(f"[鐘慢測試] 理論預期值:   {expected_proper_time}")
    
    # 6. 驗證
    assert sp.simplify(proper_time - expected_proper_time) == 0, \
        "計算出的原時不符合鐘慢效應公式"
    print("-> [PASS] 鐘慢效應驗證成功。")


# --------------------------------------------------
# 尺縮效應 (Length Contraction) 的幾何驗證
# AI 解說: https://gemini.google.com/share/e4bc5d4031fd
# --------------------------------------------------
def test_length_contraction():
    r"""
    ### 🧪 驗證狹義相對論：尺縮效應 (Length Contraction)
    
    **物理原理**:
    測量一個正在移動的物體長度。
    假設尺的靜止長度 (Proper Length) 為 $L_0$。
    當它以速度 $v$ 相對於觀察者移動時，觀察者同時測量兩端點得到的長度 $L$ 應為：
    $$ L = L_0 \sqrt{1 - v^2/c^2} = \frac{L_0}{\gamma} $$
    
    **幾何驗證方法**:
    利用勞倫茲變換 (Lorentz Transformation) 連接兩個參考系。
    """
    # 1. 初始化
    metric = get_minkowski_metric()
    # 這裡我們不直接使用 metric.arc_length，而是使用 SymPy 驗證座標變換導致的距離差異
    
    v = sp.Symbol('v', real=True)
    L_measured = sp.Symbol('L', real=True, positive=True) # 實驗室測到的長度
    gamma = 1 / sp.sqrt(1 - v**2) # 勞倫茲因子 (c=1)
    
    # 2. 定義事件 (在實驗室參考系 Lab Frame)
    # 我們在實驗室時間 t=0 "同時" 測量尺的頭尾
    # 事件 A (尺尾): x = 0, t = 0
    # 事件 B (尺頭): x = L_measured, t = 0
    # y, z 均為 0
    
    # 3. 變換到尺的靜止參考系 (Rest Frame, primed coordinates)
    # 使用勞倫茲變換:
    # x' = gamma * (x - v*t)
    # t' = gamma * (t - v*x)
    
    # 事件 A 在靜止系座標:
    x_prime_A = gamma * (0 - v * 0)
    
    # 事件 B 在靜止系座標:
    x_prime_B = gamma * (L_measured - v * 0)
    
    # 4. 計算靜止長度 (Proper Length) L_0
    # 在尺的靜止系中，尺是不動的，所以兩端點的空間距離就是靜止長度 L_0
    # (注意：雖然 t'_A 和 t'_B 不同，但在靜止系中尺不動，所以任何時間測量 x' 都是一樣的)
    L_proper_calculated = x_prime_B - x_prime_A
    
    # 5. 驗證尺縮公式: L_measured = L_proper / gamma
    # 即驗證: L_proper = L_measured * gamma
    
    print(f"\n[尺縮測試] 實驗室測量長度: {L_measured}")
    print(f"[尺縮測試] 推導出的靜止長度 (L_0): {sp.simplify(L_proper_calculated)}")
    print(f"[尺縮測試] 預期關係 (L_0 = L * gamma): {L_measured * gamma}")
    
    diff = sp.simplify(L_proper_calculated - L_measured * gamma)
    
    assert diff == 0, \
        "座標變換後的長度關係不符合尺縮效應"
    print("-> [PASS] 尺縮效應驗證成功。")

# --------------------------------------------------
# 雙生子佯謬 (Twin Paradox) 的路徑積分驗證
# AI 解說: https://gemini.google.com/share/0c63b35dea3c
# --------------------------------------------------
def test_twin_paradox_path_integral():
    r"""
    ### 🧪 驗證：雙生子佯謬 (路徑積分比較)
    
    比較兩條連接相同時空點 (Event 1 -> Event 2) 的路徑原時：
    1. 慣性觀察者 (地球上的哥哥): 直線路徑
    2. 旅行觀察者 (太空中的弟弟): 折線路徑 (飛出去再飛回來)
    
    預期結果: 慣性路徑的原時最長 (弟弟比較年輕)。
    """
    metric = get_minkowski_metric()
    t, x, y, z = metric.coords
    v = sp.Symbol('v', real=True, positive=True) # 速度 0 < v < 1
    T = sp.Symbol('T', real=True, positive=True) # 單程座標時間
    
    # 路徑 1: 哥哥 (Stay at home)
    # t 從 0 到 2T, x = 0
    path_home = [t, 0, 0, 0]
    tau_home = metric.arc_length(path_home, t, 0, 2*T)
    
    # 路徑 2: 弟弟 (Traveling)
    # 去程: t 從 0 到 T, x = v*t
    # 回程: t 從 T 到 2T, x = v*(2T - t)  (速度為 -v)
    
    # 由於 metric.arc_length 處理分段函數較複雜，我們分兩段積分相加
    # 去程
    path_out = [t, v*t, 0, 0]
    tau_out = metric.arc_length(path_out, t, 0, T)
    
    # 回程 (速度平方仍為 v^2，故積分結果結構相同，這裡直接利用對稱性或重新計算)
    # ds^2 = dt^2 - (-v dt)^2 = (1-v^2) dt^2
    path_back = [t, v*(2*T - t), 0, 0]
    tau_back = metric.arc_length(path_back, t, T, 2*T)
    
    tau_traveler = sp.simplify(tau_out + tau_back)
    
    print(f"\n[雙生子測試] 居家者原時: {tau_home}") # 應該是 2T
    print(f"[雙生子測試] 旅行者原時: {tau_traveler}") # 應該是 2T * sqrt(1-v^2)
    
    # 驗證居家者變老得比較快 (tau_home > tau_traveler)
    # 即驗證 ratio < 1
    ratio = sp.simplify(tau_traveler / tau_home)
    expected_ratio = sp.sqrt(1 - v**2)
    
    assert sp.simplify(ratio - expected_ratio) == 0, "雙生子原時比例計算錯誤"
    
    # 數值驗證: 假設 v = 0.6c (gamma = 1.25)
    # 旅行者時間應為居家者的 0.8 倍
    val = ratio.subs(v, 0.6)
    assert abs(val - 0.8) < 1e-9, "數值驗證失敗"
    
    print("-> [PASS] 雙生子佯謬路徑積分驗證成功 (慣性系原時最長)。")