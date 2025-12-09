import sympy as sp
# --------------------------------------------------
# II. Metric 類別 (度規數據封裝)
# --------------------------------------------------

class Metric:
    """
    通用度規類別，封裝度規張量 g_ij 及其逆張量 g^ij。
    """
    
    def __init__(self, g_matrix, coords):
        if sp.Matrix(g_matrix).shape != (len(coords), len(coords)):
            raise ValueError("度規矩陣的維度必須與坐標數量一致。")
            
        self.g = sp.Matrix(g_matrix)
        self.g_inv = self.g.inv()
        self.coords = coords
        self.dim = len(coords)
        self.det_g = self.g.det()
        # 使用 sp.Abs 確保行列式為非負數 (雖然對於閔可夫斯基度規，其行列式為負，
        # 但在計算體積元素時我們需要其絕對值的平方根，即 sqrt(|det(g)|))
        self.sqrt_det_g = sp.sqrt(sp.Abs(self.det_g))

# --------------------------------------------------
# I. 基礎定義：直角坐標系 (歐幾里得度規)
# --------------------------------------------------

# 宣告直角坐標變數
x, y, z = sp.symbols('x y z')
euclidean_coords = [x, y, z]

# 歐幾里得度規矩陣 g_ij = diag(1, 1, 1)
euclidean_g_matrix = sp.Matrix([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])
# 建立預設的歐幾里得度規實例
EUCLIDEAN_METRIC = Metric(euclidean_g_matrix, euclidean_coords)

# --------------------------------------------------
# III. 球坐標系 (歐幾里得空間)
# --------------------------------------------------

# 宣告球坐標變數
r, theta, phi = sp.symbols('r theta phi')
spherical_coords = [r, theta, phi]

# 球坐標度規矩陣 g_ij
# g_rr = 1, g_tt = r^2, g_pp = r^2 * sin(theta)^2
spherical_g_matrix = sp.Matrix([
    [1, 0, 0],
    [0, r**2, 0],
    [0, 0, r**2 * sp.sin(theta)**2]
])
# 建立球坐標度規實例
SPHERICAL_METRIC = Metric(spherical_g_matrix, spherical_coords)

# --------------------------------------------------
# IV. 閔可夫斯基度規 (狹義相對論)
# --------------------------------------------------

# 宣告時空坐標變數
t, x_m, y_m, z_m = sp.symbols('t x_m y_m z_m') # 使用不同的變數名稱以避免衝突
minkowski_coords = [t, x_m, y_m, z_m]

# 閔可夫斯基度規矩陣 eta_munu (採用 (+, -, -, -) 約定)
minkowski_g_matrix = sp.Matrix([
    [1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, -1, 0],
    [0, 0, 0, -1]
])
# 建立閔可夫斯基度規實例
MINKOWSKI_METRIC = Metric(minkowski_g_matrix, minkowski_coords)