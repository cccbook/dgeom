import sympy as sp
from ._metric import MetricTensor, GeometricTensor

class Spacetime:
    """
    廣義相對論的時空物件 (Spacetime Manifold)。
    
    它 '擁有' 一個度規 (Metric)，並負責計算物理量
    (如愛因斯坦張量、能量動量張量關係等)。
    """
    
    def __init__(self, metric: MetricTensor, name="Unknown Spacetime"):
        if not isinstance(metric, MetricTensor):
            raise TypeError("Spacetime 必須基於一個 MetricTensor。")
            
        self.metric = metric
        self.coords = metric.coords
        self.dim = metric.dim
        self.name = name
        
        # 快取
        self._einstein = None

    def __repr__(self):
        return f"<Spacetime: {self.name} (Dim={self.dim})>"

    # ----------------------------------------------------
    # 物理計算
    # ----------------------------------------------------

    def einstein_tensor(self):
        """
        計算愛因斯坦張量 G_{uv} = R_{uv} - 1/2 R g_{uv}
        """
        if self._einstein is not None:
            return self._einstein
            
        # 委託 metric 計算幾何量
        Ricci = self.metric.ricci_tensor()
        R_scalar = self.metric.ricci_scalar()
        g = self.metric.data
        
        # 計算物理量 G
        G_data = Ricci.data - sp.Rational(1, 2) * R_scalar * g
        
        # 包裝回幾何張量
        self._einstein = GeometricTensor(sp.simplify(G_data), self.coords, [-1, -1])
        return self._einstein

    def field_equations(self, T_uv=None, kappa=8*sp.pi):
        """
        計算愛因斯坦場方程式誤差 E = G - kappa * T
        """
        G = self.einstein_tensor()
        
        if T_uv is None:
            return G # 真空: G = 0
            
        # 處理 T_uv 輸入
        if hasattr(T_uv, 'data'):
            T_data = T_uv.data
        else:
            # 假設輸入是列表或矩陣
            T_data = sp.MutableDenseNDimArray(T_uv)

        E_data = G.data - kappa * T_data
        return GeometricTensor(sp.simplify(E_data), self.coords, [-1, -1])

    # ----------------------------------------------------
    # 代理方法 (Delegation) - 方便直接存取幾何性質
    # ----------------------------------------------------
    
    def christoffel_symbols(self):
        return self.metric.christoffel_symbols()
        
    def riemann_tensor(self):
        return self.metric.riemann_tensor()

    def ricci_scalar(self):
        return self.metric.ricci_scalar()

# ===================================================================
# 1. 狹義相對論 (Special Relativity)
# ===================================================================

def minkowski_metric():
    """
    建立閔可夫斯基時空 (Minkowski Spacetime)。
    描述: 平直時空，無重力場。
    
    坐標: [t, x, y, z]
    簽名: (+, -, -, -)
    """
    # 定義符號
    t = sp.Symbol('t', real=True)
    x, y, z = sp.symbols('x y z', real=True)
    coords = [t, x, y, z]
    
    # 度規矩陣: diag(1, -1, -1, -1)
    # 這裡假設 t 為時間單位 (c=1 的自然單位制，或 x^0 = ct)
    g_matrix = sp.diag(1, -1, -1, -1)
    
    # 1. 建立幾何層 (Math)
    tm = MetricTensor(g_matrix, coords)
    
    # 2. 回傳物理層 (Physics)
    return Spacetime(tm, name="Minkowski")

# ===================================================================
# 2. 史瓦西解 (Schwarzschild Solution)
# ===================================================================

def schwarzschild_metric():
    """
    建立史瓦西時空 (Schwarzschild Spacetime)。
    描述: 靜態、球對稱、真空中的重力場 (如黑洞或恆星外部)。
    
    坐標: [t, r, theta, phi] (球坐標)
    參數: G (重力常數), M (質量), c (光速)
    """
    # 定義符號 (使用 positive=True 幫助 SymPy 簡化 sqrt 和 sign)
    t, r, theta, phi = sp.symbols('t r theta phi', real=True)
    G, M, c = sp.symbols('G M c', real=True, positive=True)
    
    # 史瓦西半徑
    R_s = 2 * G * M / c**2
    
    # 度規因子
    # 為了避免除以零，通常假設 r > R_s 或處理視界問題，但符號運算沒關係
    factor = 1 - R_s / r
    
    # 度規矩陣 (+, -, -, -)
    # ds^2 = c^2(1-Rs/r)dt^2 - (1-Rs/r)^-1 dr^2 - r^2 dtheta^2 - ...
    g_matrix = sp.diag(
        c**2 * factor,      # g_tt
        -1 / factor,        # g_rr
        -r**2,              # g_theta_theta
        -r**2 * sp.sin(theta)**2  # g_phi_phi
    )
    
    coords = [t, r, theta, phi]
    
    tm = MetricTensor(g_matrix, coords)
    return Spacetime(tm, name="Schwarzschild")

# ===================================================================
# 3. FLRW 宇宙學模型 (Friedmann–Lemaître–Robertson–Walker)
# ===================================================================

def flrw_metric(k_val=None):
    """
    建立 FLRW 時空。
    描述: 均勻且各向同性的膨脹宇宙。
    
    坐標: [t, r, theta, phi]
    參數: 
        c: 光速
        a(t): 標度因子 (Scale Factor)
        k: 空間曲率 (-1, 0, +1)
        
    Args:
        k_val: 若提供數值 (如 0, 1, -1) 則代入，否則使用符號 k。
    """
    t, r, theta, phi = sp.symbols('t r theta phi', real=True)
    c = sp.symbols('c', real=True, positive=True)
    
    # a 是時間 t 的函數
    a = sp.Function('a')(t)
    
    # 處理曲率常數 k
    if k_val is not None:
        k = k_val
    else:
        k = sp.Symbol('k', real=True)
        
    # 徑向分量因子
    # dr^2 / (1 - k r^2)
    radial_factor = 1 / (1 - k * r**2)
    
    # 度規矩陣
    # ds^2 = c^2 dt^2 - a(t)^2 [ dr^2/(1-kr^2) + r^2 dOmega^2 ]
    g_matrix = sp.diag(
        c**2,                       # g_tt
        -a**2 * radial_factor,      # g_rr
        -a**2 * r**2,               # g_theta_theta
        -a**2 * r**2 * sp.sin(theta)**2   # g_phi_phi
    )
    
    coords = [t, r, theta, phi]
    
    tm = MetricTensor(g_matrix, coords)
    
    # 名稱加上 k 的狀態以便識別
    name = f"FLRW (k={k_val if k_val is not None else 'sym'})"
    return Spacetime(tm, name=name)

# ===================================================================
# 4. (進階) 克爾度規 (Kerr Metric) - 旋轉黑洞
# ===================================================================

def kerr_metric():
    """
    建立克爾時空 (Kerr Spacetime)。
    描述: 旋轉、軸對稱的真空黑洞解。使用 Boyer-Lindquist 坐標。
    注意: 此度規非常複雜，計算曲率張量會消耗大量時間與記憶體。
    
    坐標: [t, r, theta, phi]
    參數: M (質量), a (角動量參數 J/Mc), c (光速), G
    """
    t, r, theta, phi = sp.symbols('t r theta phi', real=True)
    G, M, c = sp.symbols('G M c', real=True, positive=True)
    a = sp.Symbol('a', real=True) # 自旋參數 (Spin parameter)
    
    # 輔助函數
    Sigma = r**2 + a**2 * sp.cos(theta)**2
    Delta = r**2 - (2*G*M*r/c**2) + a**2
    
    # 為了簡潔，常將 2GMr/c^2 寫作 rs * r，這裡展開寫
    Rs = 2*G*M/c**2
    
    # Kerr Metric 分量 (Boyer-Lindquist)
    # 參考: Visser, "The Kerr spacetime: A brief introduction"
    # ds^2 = - (1 - 2Mr/Sigma) dt^2 
    #        - (4Mar sin^2(theta) / Sigma) dt dphi 
    #        + (Sigma / Delta) dr^2 
    #        + Sigma dtheta^2 
    #        + (r^2 + a^2 + 2Ma^2r sin^2(theta)/Sigma) sin^2(theta) dphi^2
    
    # 注意: SymPy 的 diag 只能建立對角矩陣，Kerr 有非對角項 (dt dphi)
    # 我們需要先建立零矩陣，再填值
    
    # 1. 建立 4x4 零矩陣 (列表形式)
    rows = 4
    cols = 4
    g_data = [[0]*cols for _ in range(rows)]
    
    # 2. 填入分量
    # g_tt
    g_data[0][0] = c**2 * (1 - Rs * r / Sigma) # 注意 c^2 因子配合 dt
    
    # g_rr
    g_data[1][1] = -Sigma / Delta
    
    # g_theta_theta
    g_data[2][2] = -Sigma
    
    # g_phi_phi
    sin2 = sp.sin(theta)**2
    A = (r**2 + a**2)**2 - Delta * a**2 * sin2
    g_data[3][3] = -(A / Sigma) * sin2
    
    # 非對角項 g_t_phi = g_phi_t
    # 係數通常包含 c，視單位制而定。這裡保持一致性
    val_t_phi = 2 * Rs * r * a * sin2 / Sigma * c # *c 平衡單位
    g_data[0][3] = val_t_phi
    g_data[3][0] = val_t_phi

    coords = [t, r, theta, phi]
    
    # 注意: 這裡傳入的是嵌套列表 (List of Lists)，MetricTensor 應該要能處理
    tm = MetricTensor(g_data, coords)
    
    return Spacetime(tm, name="Kerr (Rotating Black Hole)")
