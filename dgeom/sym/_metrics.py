# dgeom/sym/metrics.py (修正版)
import sympy as sp
# --------------------------------------------------
# II. Metric 類別 (度規數據封裝)
# --------------------------------------------------

class Metric:
    # ... (Metric 類別定義不變) ...
    def __init__(self, g_matrix, coords):
        if sp.Matrix(g_matrix).shape != (len(coords), len(coords)):
            raise ValueError("度規矩陣的維度必須與坐標數量一致。")
            
        self.g = sp.Matrix(g_matrix) # 度規張量 g_ij
        self.g_inv = self.g.inv() # 逆度規張量 g^ij
        self.coords = coords # 坐標變數列表
        self.dim = len(coords) # 度規維度
        self.det_g = self.g.det() # 度規行列式
        self.sqrt_det_g = sp.sqrt(sp.Abs(self.det_g)) # 度規行列式的平方根

# --------------------------------------------------
# I. 基礎符號與常數宣告 (單一事實來源)
# --------------------------------------------------

# 歐幾里得直角坐標 (3D)
x, y, z = sp.symbols('x y z')
euclidean_coords = [x, y, z]

# 球坐標/時空坐標 (4D: t, r, theta, phi)
t, r, theta, phi = sp.symbols('t r theta phi')
spherical_coords = [r, theta, phi] # 3D 空間部分

# 狹義相對論空間坐標 (若與直角坐標不同，則單獨宣告)
x_m, y_m, z_m = sp.symbols('x_m y_m z_m')
minkowski_coords = [t, x_m, y_m, z_m]

# 宇宙學參數與常數
M, G, c = sp.symbols('M G c', real=True, positive=True) # 質量、萬有引力常數、光速
k = sp.symbols('k', integer=True) # 空間曲率 k

# --------------------------------------------------
# I. 歐幾里得度規實例
# --------------------------------------------------
euclidean_g_matrix = sp.Matrix([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])
EUCLIDEAN_METRIC = Metric(euclidean_g_matrix, euclidean_coords)

# --------------------------------------------------
# III. 球坐標系實例
# --------------------------------------------------
spherical_g_matrix = sp.Matrix([
    [1, 0, 0],
    [0, r**2, 0],
    [0, 0, r**2 * sp.sin(theta)**2]
])
SPHERICAL_METRIC = Metric(spherical_g_matrix, spherical_coords)

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
# V. 圓柱坐標系 (略，保持不變)
# --------------------------------------------------
rho, phi_cyl, z_cyl = sp.symbols(r'\rho \phi_c z_c') 
cylindrical_coords = [rho, phi_cyl, z_cyl]
cylindrical_g_matrix = sp.Matrix([
    [1, 0, 0],
    [0, rho**2, 0],
    [0, 0, 1]
])
CYLINDRICAL_METRIC = Metric(cylindrical_g_matrix, cylindrical_coords)

# --------------------------------------------------
# VI. 極坐標系 (略，保持不變)
# --------------------------------------------------
r_polar, theta_polar = sp.symbols(r'r_p \theta_p') 
polar_coords = [r_polar, theta_polar]
polar_g_matrix = sp.Matrix([
    [1, 0],
    [0, r_polar**2]
])
POLAR_METRIC = Metric(polar_g_matrix, polar_coords)

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
SCHWARZSCHILD_METRIC = Metric(schwarzschild_g_matrix, schwarzschild_coords)

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
FLRW_METRIC = Metric(flrw_g_matrix, flrw_coords)