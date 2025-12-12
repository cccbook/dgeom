from ._metrics import *
import sympy as sp

class RelativityMetric(Metric):
    # --------------------------------------------------
    # 新增功能: 愛因斯坦張量與重力場方程式
    # --------------------------------------------------

    def einstein_tensor(self):
        """
        計算愛因斯坦張量 (Einstein Tensor): G_{ij}。
        G_{ij} = R_{ij} - (1/2) * R * g_{ij}
        
        回傳:
            一個 sympy.Matrix 代表 Einstein Tensor。
        """
        # 檢查是否已計算過，避免重複運算
        if hasattr(self, '_einstein') and self._einstein is not None:
            return self._einstein
            
        Ricci = self.ricci_tensor()
        R_scalar = self.ricci_scalar()
        
        # 根據定義計算 G_{uv}
        # 注意: sp.Rational(1, 2) 用於確保分數運算的精確性
        G = Ricci - sp.Rational(1, 2) * R_scalar * self.g
        
        self._einstein = sp.simplify(G)
        return self._einstein

    def gravity_field_equations(self, T_uv=None, kappa=8*sp.pi):
        """
        計算愛因斯坦重力場方程式 (EFE)。
        標準形式: G_{uv} = kappa * T_{uv} (忽略宇宙常數 Lambda)
        計算結果: E_{uv} = G_{uv} - kappa * T_{uv}
        
        參數:
            T_uv: 能量-動量張量 (Matrix), 預設為 None (代表真空 Vacuum)。
            kappa: 耦合常數 (預設 8*pi，視單位制而定，SI制常為 8*pi*G/c^4)。
        
        回傳:
            一個 Matrix。若結果為零矩陣，代表滿足場方程式。
        """
        G_uv = self.einstein_tensor()
        
        if T_uv is None:
            # 真空解: G_{uv} = 0
            return G_uv
        else:
            # 非真空: G_{uv} - kappa * T_{uv} = 0
            # 回傳差值，若為 0 則滿足方程
            return sp.simplify(G_uv - kappa * T_uv)

def get_minkowski_metric():
    """
    建立 4D 閔可夫斯基度規 (狹義相對論, SR)。
    
    簽名慣例: (+, -, -, -)
    度規形式: diag(1, -1, -1, -1)
    
    注意: 此處 g_tt = 1，隱含了使用的是自然單位制 (c=1) 
          或者是將時間坐標定義為 x^0 = ct。
    """
    # 局部定義符號，避免與全域變數衝突
    t = sp.Symbol('t', real=True)
    x_m, y_m, z_m = sp.symbols('x_m y_m z_m', real=True)
    
    coords = [t, x_m, y_m, z_m]
    
    # 根據您提供的矩陣定義
    g_matrix = sp.diag(1, -1, -1, -1)
    
    return RelativityMetric(g_matrix, coords)

def get_schwarzschild_metric():
    """
    建立史瓦西度規 (Schwarzschild Metric)。
    坐標: t, r, theta, phi
    常數: G, M, c
    """
    t, r, theta, phi = sp.symbols('t r theta phi')
    G, M, c = sp.symbols('G M c', real=True, positive=True)
    
    R_s = 2 * G * M / c**2
    f_r = 1 - R_s / r
    
    # 簽名慣例 (+, -, -, -)
    g_matrix = sp.diag(
        c**2 * f_r,
        -1 / f_r,
        -r**2,
        -r**2 * sp.sin(theta)**2
    )
    
    coords = [t, r, theta, phi]
    return RelativityMetric(g_matrix, coords)

def get_flrw_metric(k_val=None):
    """
    建立 FLRW 度規 (宇宙學)。
    坐標: t, r, theta, phi
    函數: a(t)
    常數: c, k (若 k_val 未指定則為符號 k)
    
    參數:
        k_val: 空間曲率常數 (如 -1, 0, 1)，若為 None 則使用符號變數。
    """
    t, r, theta, phi = sp.symbols('t r theta phi')
    c = sp.symbols('c', real=True, positive=True)
    a = sp.Function('a')(t)
    
    if k_val is None:
        k = sp.symbols('k', real=True)
    else:
        k = k_val
        
    D_r = 1 / (1 - k * r**2)
    
    # 簽名慣例 (+, -, -, -)
    g_matrix = sp.diag(
        c**2,
        -a**2 * D_r,
        -a**2 * r**2,
        -a**2 * r**2 * sp.sin(theta)**2
    )
    
    coords = [t, r, theta, phi]
    return RelativityMetric(g_matrix, coords)

SCHWARZSCHILD_METRIC = get_schwarzschild_metric()
FLRW_METRIC = get_flrw_metric()
MINKOWSKI_METRIC = get_minkowski_metric()
