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

# 狹義相對論空間坐標 (若與直角坐標不同，則單獨宣告)
x_m, y_m, z_m = sp.symbols('x_m y_m z_m')
minkowski_coords = [t, x_m, y_m, z_m]
# t = sp.symbols('t r theta phi', real=True, positive=True)

# 宇宙學參數與常數
M, G, c = sp.symbols('M G c', real=True, positive=True) # 質量、萬有引力常數、光速
k = sp.symbols('k', integer=True) # 空間曲率 k

# --------------------------------------------------
# IV. 閔可夫斯基度規實例
# --------------------------------------------------
minkowski_g_matrix = sp.Matrix([
    [1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, -1, 0],
    [0, 0, 0, -1]
])
MINKOWSKI_METRIC = Metric(minkowski_g_matrix, minkowski_coords)

# --------------------------------------------------
# VII. 施瓦西度規實例
# --------------------------------------------------
# 使用統一後的符號 t, r, theta, phi
schwarzschild_coords = [t, r, theta, phi] 
R_s = 2 * G * M / c**2
f_r = 1 - R_s / r
f_inv_r = 1 / f_r
r_sq = r**2
sin_sq_theta = sp.sin(theta)**2
schwarzschild_g_matrix = sp.Matrix([
    [c**2 * f_r, 0, 0, 0],
    [0, -f_inv_r, 0, 0],
    [0, 0, -r_sq, 0],
    [0, 0, 0, -r_sq * sin_sq_theta]
])
SCHWARZSCHILD_METRIC = RelativityMetric(schwarzschild_g_matrix, schwarzschild_coords)


"""
# --------------------------------------------------
# VIII. FLRW 度規實例
# --------------------------------------------------
# 使用統一後的符號 t, r, theta, phi
flrw_coords = [t, r, theta, phi] 
a_t = sp.Function('a')(t) # 宇宙標度因子 a(t)
D_r = 1 / (1 - k * r**2)
flrw_g_matrix = sp.Matrix([
    [c**2, 0, 0, 0],
    [0, -a_t**2 * D_r, 0, 0],
    [0, 0, -a_t**2 * r**2, 0],
    [0, 0, 0, -a_t**2 * r**2 * sp.sin(theta)**2]
])
FLRW_METRIC = RelativityMetric(flrw_g_matrix, flrw_coords)

"""

# 注意: k 是曲率常數 (-1, 0, +1)，這裡設為整數或實數皆可
k = sp.symbols('k', real=True) 
c = sp.symbols('c', real=True, positive=True)

# 定義宇宙標度因子 a(t) 為時間的函數
a = sp.Function('a')(t)

# 對時間的導數符號 (用於顯示更漂亮的輸出)
adot = sp.diff(a, t)
addot = sp.diff(a, t, t)

coords = [t, r, theta, phi]

# --------------------------------------------------
# 2. 定義 FLRW 度規矩陣
# --------------------------------------------------
# 根據您的輸入: g_tt = c^2 (簽名 + - - -)
# g_rr = -a(t)^2 / (1 - k*r^2)
D_r = 1 / (1 - k * r**2)

flrw_g_matrix = sp.Matrix([
    [c**2, 0, 0, 0],
    [0, -a**2 * D_r, 0, 0],
    [0, 0, -a**2 * r**2, 0],
    [0, 0, 0, -a**2 * r**2 * sp.sin(theta)**2]
])

print("初始化 FLRW Metric 物件...")
# 假設類別名稱為 Metric，若您的類別名為 RelativityMetric 請自行修改
FLRW_METRIC = RelativityMetric(flrw_g_matrix, coords)
