import sympy as sp
import numpy as np
from sympy import MutableDenseNDimArray, Matrix, diff
from ._tensor import GeometricTensor

class MetricTensor(GeometricTensor):
    """
    度規張量 (Metric Tensor)。
    
    特徵：
    1. 固定為 Rank 2 協變張量 (Type=[-1, -1])
    2. 提供幾何計算方法 (Inverse, Christoffel, Riemann, Geodesics)
    """

    def __init__(self, data, coords):
        # 強制設定為協變 Rank 2
        super().__init__(data, coords, [-1, -1])
        
        # 驗證方陣
        rows, cols = self.data.shape
        if rows != cols:
            raise ValueError("度規張量必須是方陣")
            
        # 驗證對稱性 (可選，但在 GR 中是必須的)
        mat = Matrix(self.data.tolist())
        if not mat.is_symmetric():
            # 僅警告，或許某些非標準理論需要非對稱度規
            print("Warning: 度規張量非對稱")

    # ==========================================
    # 核心幾何量 (Geometric Objects)
    # ==========================================

    def inverse(self):
        """計算逆度規 g^uv (Type=[1, 1])"""
        g_mat = Matrix(self.data.tolist())
        g_inv_mat = g_mat.inv()
        return GeometricTensor(MutableDenseNDimArray(g_inv_mat.tolist()), self.coords, [1, 1])

    def christoffel_symbols(self):
        """
        計算第二類克里斯多福符號 Gamma^k_ij (Type=[1, -1, -1])
        """
        dim = self.dim
        coords = self.coords
        g_inv = self.inverse() # [1, 1]
        
        # 預計算導數 d_k g_ij
        # partial_g[k][i][j]
        partial_g = [[[diff(self.data[i, j], x_k) for j in range(dim)] 
                      for i in range(dim)] 
                     for x_k in coords]

        gamma_data = sp.MutableDenseNDimArray.zeros(dim, dim, dim)

        for k in range(dim):
            for i in range(dim):
                for j in range(dim):
                    val = 0
                    # Gamma^k_ij = 0.5 * g^kl * (d_j g_il + d_i g_jl - d_l g_ij)
                    for l in range(dim):
                        term = partial_g[j][i][l] + partial_g[i][j][l] - partial_g[l][i][j]
                        val += g_inv.data[k, l] * term
                    
                    gamma_data[k, i, j] = sp.simplify(0.5 * val)
                    
        return GeometricTensor(gamma_data, coords, [1, -1, -1])

    def riemann_tensor(self):
        """
        計算黎曼曲率張量 R^rho_sigma,mu,nu (Type=[1, -1, -1, -1])
        """
        gamma = self.christoffel_symbols() # Type=[1, -1, -1]
        G = gamma.data
        dim = self.dim
        coords = self.coords
        
        R_data = sp.MutableDenseNDimArray.zeros(dim, dim, dim, dim)
        
        for rho in range(dim):
            for sigma in range(dim):
                for mu in range(dim):
                    for nu in range(dim):
                        # term1: d_mu Gamma^rho_nu,sigma
                        t1 = diff(G[rho, nu, sigma], coords[mu])
                        # term2: d_nu Gamma^rho_mu,sigma
                        t2 = diff(G[rho, mu, sigma], coords[nu])
                        
                        # term3, term4: quadratic terms
                        t3 = 0
                        t4 = 0
                        for lam in range(dim):
                            t3 += G[rho, mu, lam] * G[lam, nu, sigma]
                            t4 += G[rho, nu, lam] * G[lam, mu, sigma]
                            
                        R_data[rho, sigma, mu, nu] = sp.simplify(t1 - t2 + t3 - t4)
                        
        return GeometricTensor(R_data, coords, [1, -1, -1, -1])

    def ricci_tensor(self):
        """
        計算里奇張量 R_mu,nu (Type=[-1, -1])
        縮併 Riemann 的第 1 (rho) 和第 3 (mu) 指標
        """
        riemann = self.riemann_tensor()
        return riemann.contract(0, 2)

    def ricci_scalar(self):
        """
        計算里奇純量 R (Type=[])
        R = g^uv R_uv
        """
        ricci = self.ricci_tensor() # [-1, -1]
        g_inv = self.inverse()      # [1, 1]
        
        # 全部縮併
        # 這裡利用 Tensor 的運算能力
        # g^uv * R_uv -> Scalar
        mixed = g_inv.tensor_product(ricci) # [1, 1, -1, -1]
        contracted = mixed.contract(0, 2).contract(0, 1) # 兩次縮併
        
        return sp.simplify(contracted.data[()])

    # ==========================================
    # 應用功能：測地線與弧長
    # ==========================================

    def arc_length(self, path_param, param_var, start_val, end_val):
        """計算路徑弧長"""
        if len(path_param) != self.dim:
            raise ValueError("路徑維度不符")
            
        dx_dt = [diff(x, param_var) for x in path_param]
        subs_rules = dict(zip(self.coords, path_param))
        
        ds_sq = 0
        for i in range(self.dim):
            for j in range(self.dim):
                g_val = sp.sympify(self.data[i, j]).subs(subs_rules)
                ds_sq += g_val * dx_dt[i] * dx_dt[j]
                
        return sp.Integral(sp.sqrt(ds_sq), (param_var, start_val, end_val))

    def get_geodesic_equations(self, param_var=sp.Symbol('tau')):
        """
        生成測地線方程: x''^k = - Gamma^k_ij x'^i x'^j
        """
        Gamma = self.christoffel_symbols()
        funcs = [sp.Function(c.name)(param_var) for c in self.coords]
        vels = [diff(f, param_var) for f in funcs]
        accs = [diff(v, param_var) for v in vels]
        
        eqs = []
        for k in range(self.dim):
            rhs = 0
            for i in range(self.dim):
                for j in range(self.dim):
                    g_val = Gamma.data[k, i, j].subs(dict(zip(self.coords, funcs)))
                    rhs += g_val * vels[i] * vels[j]
            
            eqs.append(sp.Eq(accs[k], sp.simplify(-rhs)))
        return eqs

    def solve_geodesic_bvp(self, start_point, end_point, num_points=50):
        """數值求解測地線 (需要 scipy)"""
        try:
            from scipy.integrate import solve_bvp
        except ImportError:
            raise ImportError("需安裝 scipy")

        Gamma = self.christoffel_symbols()
        gamma_func = sp.lambdify(self.coords, Gamma.data.tolist(), modules='numpy')
        dim = self.dim

        def ode_system(t, y):
            pos = y[:dim]
            vel = y[dim:]
            acc = np.zeros_like(vel)
            
            # Vectorized implementation for speed
            # G shape: (dim, dim, dim, num_points)
            # 這裡簡單用 loop 處理每個時間點
            for i in range(pos.shape[1]):
                p = pos[:, i]
                v = vel[:, i]
                G = np.array(gamma_func(*p))
                # a^k = - G^k_ij v^i v^j
                acc[:, i] = -np.einsum('kij,i,j->k', G, v, v)
                
            return np.vstack((vel, acc))

        def bc(ya, yb):
            return np.concatenate((ya[:dim] - start_point, yb[:dim] - end_point))

        # Initial guess: linear path
        x = np.linspace(0, 1, num_points)
        y_guess = np.zeros((2*dim, num_points))
        for i in range(dim):
            y_guess[i] = np.linspace(start_point[i], end_point[i], num_points)
            y_guess[dim+i] = end_point[i] - start_point[i]

        res = solve_bvp(ode_system, bc, x, y_guess, tol=1e-3)
        return res.y[:dim]

# ==========================================
# 工廠函數 (Factory Functions)
# ==========================================

def euclidean_metric():
    x, y, z = sp.symbols('x y z')
    return MetricTensor(sp.eye(3), [x, y, z])

def spherical_metric():
    r, theta, phi = sp.symbols('r theta phi', real=True, positive=True)
    g = sp.diag(1, r**2, r**2 * sp.sin(theta)**2)
    return MetricTensor(g, [r, theta, phi])

def cylindrical_metric():
    rho, phi, z = sp.symbols(r'\rho \phi z', real=True, positive=True)
    g = sp.diag(1, rho**2, 1)
    return MetricTensor(g, [rho, phi, z])

def polar_metric():
    r, theta = sp.symbols('r theta', real=True, positive=True)
    g = sp.diag(1, r**2)
    return MetricTensor(g, [r, theta])
