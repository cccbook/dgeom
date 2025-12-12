import sympy as sp
from sympy.tensor.array import MutableDenseNDimArray

# 度規 metrics
# AI 解說： https://gemini.google.com/share/a3c23e660d10

class Metric:
    def __init__(self, g_matrix, coords):
        if sp.Matrix(g_matrix).shape != (len(coords), len(coords)):
            raise ValueError("度規矩陣的維度必須與坐標數量一致。")
            
        self.g = sp.Matrix(g_matrix) # 度規張量 g_ij
        self.g_inv = self.g.inv() # 逆度規張量 g^ij
        self.coords = coords # 坐標變數列表
        self.dim = len(coords) # 度規維度
        self.det_g = self.g.det() # 度規行列式
        self.sqrt_det_g = sp.sqrt(sp.Abs(self.det_g)) # 度規行列式的平方根
        
        # 初始化緩存屬性
        self._christoffel = None
        self._riemann = None
        self._ricci = None
        self._ricci_scalar = None
        
    def christoffel_symbols(self):
        """
        計算克里斯多福符號 (第二類): Gamma^k_{ij}。
        Gamma^k_{ij} = g^{kl} * Gamma_{l, ij}
        
        回傳:
            一個 sympy.MutableDenseNDimArray 代表 Christoffel 符號。
        """
        if self._christoffel is not None:
            return self._christoffel
        
        Gamma = MutableDenseNDimArray.zeros(self.dim, self.dim, self.dim)

        # 第一類克里斯多福符號 Gamma_{l, i, j}
        # 公式: Gamma_{l, i, j} = 0.5 * (g_{il, j} + g_{jl, i} - g_{ij, l})
        Gamma_first_kind = MutableDenseNDimArray.zeros(self.dim, self.dim, self.dim)
        for l in range(self.dim): # l: 降低的指標
            for i in range(self.dim):
                for j in range(self.dim):
                    g_il_j = sp.diff(self.g[i, l], self.coords[j])
                    g_jl_i = sp.diff(self.g[j, l], self.coords[i])
                    g_ij_l = sp.diff(self.g[i, j], self.coords[l])
                    
                    Gamma_first_kind[l, i, j] = sp.simplify(sp.Rational(1, 2) * (g_il_j + g_jl_i - g_ij_l))
        
        # 第二類克里斯多福符號: Gamma^k_{ij} = g^{kl} * Gamma_{l, ij}
        for k in range(self.dim):
            for i in range(self.dim):
                for j in range(self.dim):
                    sum_l = 0
                    for l in range(self.dim):
                        sum_l += self.g_inv[k, l] * Gamma_first_kind[l, i, j] 
                    Gamma[k, i, j] = sp.simplify(sum_l)

        self._christoffel = Gamma
        return Gamma
        
    def riemann_tensor(self):
        """
        計算黎曼曲率張量 (上一下三的混合型式): R^k_{lij}。
        R^k_{lij} = \partial_i \Gamma^k_{jl} - \partial_j \Gamma^k_{il} + \Gamma^m_{jl} \Gamma^k_{im} - \Gamma^m_{il} \Gamma^k_{jm}
        
        回傳:
            一個 sympy.MutableDenseNDimArray 代表 Riemann 張量。
        """
        if self._riemann is not None:
            return self._riemann
            
        Gamma = self.christoffel_symbols()
        R = MutableDenseNDimArray.zeros(self.dim, self.dim, self.dim, self.dim)
        
        for k in range(self.dim):
            for l in range(self.dim):
                for i in range(self.dim):
                    for j in range(self.dim):
                        # 項 I: \partial_i \Gamma^k_{jl}
                        term1 = sp.diff(Gamma[k, j, l], self.coords[i])
                        
                        # 項 II: - \partial_j \Gamma^k_{il}
                        term2 = -sp.diff(Gamma[k, i, l], self.coords[j])
                        
                        # 項 III: \Gamma^m_{jl} \Gamma^k_{im}
                        sum3 = 0
                        for m in range(self.dim):
                            sum3 += Gamma[m, j, l] * Gamma[k, i, m]
                            
                        # 項 IV: - \Gamma^m_{il} \Gamma^k_{jm}
                        sum4 = 0
                        for m in range(self.dim):
                            sum4 += Gamma[m, i, l] * Gamma[k, j, m]
                        
                        R[k, l, i, j] = sp.simplify(term1 + term2 + sum3 - sum4)

        self._riemann = R
        return R

    def ricci_tensor(self):
        """
        計算里奇張量 (Ricci Tensor): R_{ij} = R^k_{ikj} (縮併第二和第四個指標)。
        
        回傳:
            一個 sympy.Matrix (二階張量) 代表 Ricci 張量。
        """
        if self._ricci is not None:
            return self._ricci
            
        R_k_lij = self.riemann_tensor()
        Ricci = sp.Matrix.zeros(self.dim, self.dim)
        
        for i in range(self.dim):
            for j in range(self.dim):
                sum_k = 0
                for k in range(self.dim):
                    # Ricci_{ij} = R^k_{ikj}
                    sum_k += R_k_lij[k, i, k, j] 
                Ricci[i, j] = sp.simplify(sum_k)
        
        self._ricci = Ricci
        return Ricci

    def ricci_scalar(self):
        """
        計算純量曲率 (Scalar Curvature): R = g^{ij} R_{ij}。
        
        回傳:
            一個 sympy 表達式 (純量) 代表 Scalar 曲率。
        """
        if self._ricci_scalar is not None:
            return self._ricci_scalar
            
        Ricci = self.ricci_tensor()
        R = 0
        
        for i in range(self.dim):
            for j in range(self.dim):
                R += self.g_inv[i, j] * Ricci[i, j]
                
        self._ricci_scalar = sp.simplify(R)
        return self._ricci_scalar

    def arc_length(self, path_param, param_var, start_param, end_param):
        """
        計算一條參數化曲線的弧長 (Arc Length)。
        L = \int_{\lambda_1}^{\lambda_2} \sqrt{g_{ij} \frac{dx^i}{d\lambda} \frac{dx^j}{d\lambda}} \, d\lambda
        
        回傳:
            符號積分表達式 (Symbolic Integral Expression)。
        """
        if len(path_param) != self.dim:
            raise ValueError("曲線坐標的數量必須與度規維度一致。")
            
        dx_dlambda = [sp.diff(x, param_var) for x in path_param]
        
        # 計算 $g_{ij} \frac{dx^i}{d\lambda} \frac{dx^j}{d\lambda}$
        integrand_sq = 0
        for i in range(self.dim):
            for j in range(self.dim):
                g_ij_on_path = self.g[i, j].subs(zip(self.coords, path_param))
                integrand_sq += g_ij_on_path * dx_dlambda[i] * dx_dlambda[j]
        
        # 弧長積分
        length = sp.Integral(sp.sqrt(integrand_sq), (param_var, start_param, end_param))
        return length

    def curvature_of_curve(self, curve_vec, param_var):
        """
        計算空間曲線的曲率 (Curvature)，假設曲線嵌入在 3D 歐幾里得空間中。
        
        回傳:
            曲率 kappa 的 sympy 表達式。
        """
        if self.dim != 3:
             raise NotImplementedError("曲線曲率/撓率方法僅在 3D (R^3) 歐幾里得嵌入空間中實現。")
             
        r_prime = sp.diff(curve_vec, param_var)
        r_double_prime = sp.diff(r_prime, param_var)
        
        # 曲率 kappa = |\vec{r}'(t) x \vec{r}''(t)| / |\vec{r}'(t)|^3
        r_prime_cross_r_double_prime = r_prime.cross(r_double_prime)
        
        # 使用點積計算向量長度的平方
        numerator = sp.sqrt(r_prime_cross_r_double_prime.dot(r_prime_cross_r_double_prime))
        denominator = sp.sqrt(r_prime.dot(r_prime))**3
        
        kappa = sp.simplify(numerator / denominator)
        return kappa

    def torsion_of_curve(self, curve_vec, param_var):
        """
        計算空間曲線的撓率 (Torsion)，假設曲線嵌入在 3D 歐幾里得空間中。
        
        回傳:
            撓率 tau 的 sympy 表達式。
        """
        if self.dim != 3:
             raise NotImplementedError("曲線曲率/撓率方法僅在 3D (R^3) 歐幾里得嵌入空間中實現。")
             
        r_prime = sp.diff(curve_vec, param_var)
        r_double_prime = sp.diff(r_prime, param_var)
        r_triple_prime = sp.diff(r_double_prime, param_var)

        # 混合積: (\vec{r}' \times \vec{r}'') \cdot \vec{r}'''
        numerator = r_prime.cross(r_double_prime).dot(r_triple_prime)
        
        # 分母: |\vec{r}' \times \vec{r}''|^2
        r_prime_cross_r_double_prime = r_prime.cross(r_double_prime)
        denominator = r_prime_cross_r_double_prime.dot(r_prime_cross_r_double_prime)
        
        tau = sp.simplify(numerator / denominator)
        return tau

# --------------------------------------------------
# Factory Functions (度規工廠函數)
# --------------------------------------------------

def get_euclidean_metric():
    """
    建立 3D 歐幾里得度規 (直角坐標)。
    坐標: x, y, z
    度規: diag(1, 1, 1)
    """
    x, y, z = sp.symbols('x y z')
    coords = [x, y, z]
    g_matrix = sp.eye(3)
    return Metric(g_matrix, coords)

def get_spherical_metric():
    """
    建立 3D 球坐標度規。
    坐標: r, theta, phi
    度規: diag(1, r^2, r^2 * sin(theta)^2)
    """
    # 物理坐標通常假設為實數且 r>0
    r, theta, phi = sp.symbols('r theta phi', real=True, positive=True)
    coords = [r, theta, phi]
    g_matrix = sp.diag(1, r**2, r**2 * sp.sin(theta)**2)
    return Metric(g_matrix, coords)

def get_cylindrical_metric():
    """
    建立 3D 圓柱坐標度規。
    坐標: rho, phi, z
    度規: diag(1, rho^2, 1)
    """
    rho, phi, z = sp.symbols(r'\rho \phi z', real=True, positive=True)
    coords = [rho, phi, z]
    g_matrix = sp.diag(1, rho**2, 1)
    return Metric(g_matrix, coords)

def get_polar_metric():
    """
    建立 2D 極坐標度規。
    坐標: r, theta
    度規: diag(1, r^2)
    """
    r, theta = sp.symbols('r theta', real=True, positive=True)
    coords = [r, theta]
    g_matrix = sp.diag(1, r**2)
    return Metric(g_matrix, coords)


