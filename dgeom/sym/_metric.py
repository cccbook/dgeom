import sympy as sp
import numpy as np
from sympy import MutableDenseNDimArray, Matrix, diff
from ._tensor import GeometricTensor
# 度規 metrics
# AI 解說： https://gemini.google.com/share/a3c23e660d10

class TensorMetric(GeometricTensor):
    """
    度規張量物件 (Metric Tensor g_{mu,nu})。
    繼承自 GeometricTensor，固定為 Rank 2 協變張量 ([-1, -1])。
    
    提供幾何計算功能：
    1. 逆度規 (Inverse Metric)
    2. 克里斯多福符號 (Christoffel Symbols)
    3. 黎曼曲率張量 (Riemann Curvature Tensor)
    """

    def __init__(self, data, coords):
        # 度規必須是 Rank 2 協變張量
        super().__init__(data, coords, [-1, -1])
        
        # 檢查是否為方陣
        rows, cols = self.data.shape
        if rows != cols:
            raise ValueError("度規張量必須是方陣。")
            
        # 檢查對稱性 (可選，但物理上度規通常是對稱的)
        mat = Matrix(self.data)
        if not mat.is_symmetric():
            # 這裡僅印出警告，某些非標準理論可能會有非對稱度規
            print("Warning: 輸入的度規張量不是對稱的。")

    def inverse(self):
        """
        計算逆度規 g^{mu,nu} (Contravariant)。
        Returns:
            GeometricTensor: Rank 2 逆變張量 ([1, 1])
        """
        # 轉成 Matrix 方便求反矩陣
        g_mat = Matrix(self.data)
        g_inv_mat = g_mat.inv()
        
        # 回傳一般的 GeometricTensor，因為逆度規不是 metric object (它是 [1, 1])
        return GeometricTensor(g_inv_mat, self.coords, [1, 1])

    def christoffel_symbols(self):
        """
        計算第二類克里斯多福符號 Gamma^k_{i,j}。
        定義: Gamma^k_{ij} = 1/2 * g^kl * (d_j g_il + d_i g_jl - d_l g_ij)
        
        Returns:
            GeometricTensor: Rank 3 混合張量 ([1, -1, -1]) -> Gamma^k_ij
        """
        coords = self.coords
        dim = self.dim
        g_inv = self.inverse() # g^kl
        
        # 準備結果容器 (Rank 3: k, i, j)
        # 注意 SymPy NDimArray 的建構通常是一維列表，這裡用 list comprehension 構建
        gamma_data = sp.MutableDenseNDimArray.zeros(dim, dim, dim)
        
        # 為了效能，預先計算所有偏導數 d_l g_ij
        # partial_g[l, i, j] = d(g_ij)/d(x^l)
        partial_g = [[[diff(self.data[i, j], x_l) for j in range(dim)] 
                      for i in range(dim)] 
                     for x_l in coords]

        for k in range(dim):
            for i in range(dim):
                for j in range(dim):
                    sum_val = 0
                    for l in range(dim):
                        # Gamma 公式
                        # term = d_j g_il + d_i g_jl - d_l g_ij
                        # 映射到我們的 partial_g 索引:
                        # d_j g_il -> partial_g[j][i][l]
                        # d_i g_jl -> partial_g[i][j][l]
                        # d_l g_ij -> partial_g[l][i][j]
                        
                        term = partial_g[j][i][l] + partial_g[i][j][l] - partial_g[l][i][j]
                        sum_val += g_inv.data[k, l] * term
                    
                    gamma_data[k, i, j] = sp.simplify(sum_val / 2)
                    
        return GeometricTensor(gamma_data, coords, [1, -1, -1])

    def riemann_tensor(self):
        """
        計算黎曼曲率張量 R^rho_{sigma, mu, nu}
        R^rho_{sigma, mu, nu} = d_mu Gamma^rho_{nu, sigma} - d_nu Gamma^rho_{mu, sigma} 
                              + Gamma^rho_{mu, lambda} Gamma^lambda_{nu, sigma}
                              - Gamma^rho_{nu, lambda} Gamma^lambda_{mu, sigma}
        Returns:
            GeometricTensor: Rank 4 張量 ([1, -1, -1, -1])
        """
        gamma = self.christoffel_symbols() # Gamma^rho_{mu, nu} -> indices (rho, mu, nu) -> (0, 1, 2)
        dim = self.dim
        coords = self.coords
        
        R_data = sp.MutableDenseNDimArray.zeros(dim, dim, dim, dim)
        
        # 為了可讀性，我們直接操作 data array
        G = gamma.data 
        
        for rho in range(dim):
            for sigma in range(dim):
                for mu in range(dim):
                    for nu in range(dim):
                        # 1. 導數項
                        # d_mu Gamma^rho_{nu, sigma}  (注意 Gamma 指標順序是 ^rho _nu _sigma) -> G[rho, nu, sigma]
                        term1 = diff(G[rho, nu, sigma], coords[mu])
                        # d_nu Gamma^rho_{mu, sigma}
                        term2 = diff(G[rho, mu, sigma], coords[nu])
                        
                        # 2. 二次項 (愛因斯坦求和 lambda)
                        term3 = 0
                        term4 = 0
                        for lam in range(dim):
                            # Gamma^rho_{mu, lambda} * Gamma^lambda_{nu, sigma}
                            term3 += G[rho, mu, lam] * G[lam, nu, sigma]
                            # Gamma^rho_{nu, lambda} * Gamma^lambda_{mu, sigma}
                            term4 += G[rho, nu, lam] * G[lam, mu, sigma]
                            
                        val = term1 - term2 + term3 - term4
                        R_data[rho, sigma, mu, nu] = sp.simplify(val)
                        
        return GeometricTensor(R_data, coords, [1, -1, -1, -1])

    def ricci_tensor(self):
        """
        計算里奇張量 (Ricci Tensor) R_{mu, nu} = R^lambda_{mu, lambda, nu}
        這是黎曼張量的縮併 (Contract 0 and 2)。
        """
        riemann = self.riemann_tensor() # indices: rho(0), sigma(1), mu(2), nu(3)
        
        # R_{sigma, nu} = Contract rho(0) and mu(2)
        # 注意: 我們的 riemann 定義通常是 R^rho_{sigma, mu, nu}
        # 縮併第一指標(上)和第三指標(下)
        
        return riemann.contract(0, 2)

    def ricci_scalar(self):
        """
        計算里奇純量 (Ricci Scalar) R = g^{mu, nu} R_{mu, nu}
        """
        ricci_t = self.ricci_tensor() # R_mn [-1, -1]
        g_inv = self.inverse()        # g^mn [1, 1]
        
        # R = g^mn R_mn
        # 先做 tensor product -> Rank 4: [1, 1, -1, -1] (indices: m_up, n_up, m_down, n_down)
        mixed = g_inv.tensor_product(ricci_t)
        
        # 兩次縮併
        # 1. Contract index 0 (m_up) and 2 (m_down)
        temp = mixed.contract(0, 2) # Remaining: n_up, n_down
        
        # 2. Contract index 0 and 1
        scalar_t = temp.contract(0, 1)
        
        # 取出純量數值
        return sp.simplify(scalar_t.data[()])

    # ==========================================
    #  新增功能：曲線與測地線
    # ==========================================

    def arc_length(self, path_param, param_var, start_val, end_val):
        """
        計算參數化曲線的弧長。
        """
        if len(path_param) != self.dim:
            raise ValueError("曲線坐標分量數量必須與維度一致。")
            
        # 計算速度向量 dx/dt
        dx_dt = [diff(x, param_var) for x in path_param]
        
        # 建立替換規則: 將坐標符號替換為參數式 (x -> x(t))
        subs_rules = dict(zip(self.coords, path_param))
        
        # 計算 ds^2 = g_ij v^i v^j
        ds_sq = 0
        for i in range(self.dim):
            for j in range(self.dim):
                # 1. 取出 metric 分量 g_ij
                g_val = self.data[i, j]
                
                # 2. 替換坐標: g_ij(x) -> g_ij(x(t))
                # 【修正】: 使用 sp.sympify 確保 int 也能被處理
                g_val_t = sp.sympify(g_val).subs(subs_rules)
                
                # 3. 累加
                ds_sq += g_val_t * dx_dt[i] * dx_dt[j]
        
        integrand = sp.sqrt(ds_sq)
        return sp.Integral(integrand, (param_var, start_val, end_val))

    def get_geodesic_equations(self, param_var=sp.Symbol('tau')):
        """
        生成測地線微分方程: x''^k + Gamma^k_ij x'^i x'^j = 0
        """
        Gamma = self.christoffel_symbols() # 取得 Gamma 張量
        
        # 定義未知函數 x(tau), y(tau)...
        funcs = [sp.Function(c.name)(param_var) for c in self.coords]
        
        # 一階導數 (速度)
        velocities = [diff(f, param_var) for f in funcs]
        # 二階導數 (加速度)
        accelerations = [diff(v, param_var) for v in velocities]
        
        equations = []
        for k in range(self.dim):
            # 計算右側: - Gamma^k_ij * u^i * u^j
            rhs = 0
            for i in range(self.dim):
                for j in range(self.dim):
                    # Gamma 裡面的坐標符號 (x, y) 必須換成函數 (x(tau), y(tau))
                    g_val = Gamma.data[k, i, j].subs(dict(zip(self.coords, funcs)))
                    rhs += g_val * velocities[i] * velocities[j]
            
            # Eq(x'', -RHS)
            eq = sp.Eq(accelerations[k], sp.simplify(-rhs))
            equations.append(eq)
            
        return equations

    def solve_geodesic_bvp(self, start_point, end_point, num_points=50):
        """
        數值求解測地線 (邊界值問題 BVP)。
        """
        try:
            from scipy.integrate import solve_bvp
        except ImportError:
            raise ImportError("需安裝 scipy: pip install scipy")

        # 1. 取得 Gamma 並轉為數值函數
        Gamma_tensor = self.christoffel_symbols()
        
        # 使用 lambdify 將符號表達式轉為 Python 函數
        # 輸入: 坐標 (x, y, ...)，輸出: Gamma 數值陣列
        # 注意: tolist() 是必要的，因為 lambdify 不總是能完美處理 NDimArray
        gamma_func = sp.lambdify(self.coords, Gamma_tensor.data.tolist(), modules='numpy')

        # 2. 定義 ODE 系統 (狀態向量 y = [pos, vel])
        dim = self.dim
        
        def ode_system(t, y_vec):
            # y_vec shape: (2*dim, num_points)
            positions = y_vec[:dim]
            velocities = y_vec[dim:]
            
            # 計算導數 dy/dt = [vel, acc]
            # acc^k = - Gamma^k_ij v^i v^j
            
            # 由於 solve_bvp 傳入的是整個網格點，我們需要對每個點計算
            # 這裡使用 list comprehension 或迴圈處理每個時間點 t
            
            res_acc = np.zeros_like(positions)
            m = positions.shape[1] # 時間點數量
            
            for i in range(m):
                pos = positions[:, i]
                vel = velocities[:, i]
                
                # 計算該位置的 Gamma 值 (dim, dim, dim)
                G_val = np.array(gamma_func(*pos))
                
                # Einstein Summation: k, i, j -> k
                # acc = - G[k, i, j] * v[i] * v[j]
                acc = -np.einsum('kij, i, j -> k', G_val, vel, vel)
                res_acc[:, i] = acc
                
            return np.vstack((velocities, res_acc))

        # 3. 邊界條件 (Residuals)
        def bc(ya, yb):
            # 起點位置誤差, 終點位置誤差
            return np.concatenate((ya[:dim] - start_point, yb[:dim] - end_point))

        # 4. 初始猜測 (直線)
        x_plot = np.linspace(0, 1, num_points)
        y_guess = np.zeros((2 * dim, num_points))
        for i in range(dim):
            y_guess[i, :] = np.linspace(start_point[i], end_point[i], num_points)
            y_guess[dim+i, :] = end_point[i] - start_point[i] # 猜測速度為常數

        # 5. 求解
        res = solve_bvp(ode_system, bc, x_plot, y_guess, tol=1e-3)
        
        if not res.success:
            print(f"BVP Solver Warning: {res.message}")
            
        return res.y[:dim] # 只回傳路徑點

    # 關於 curvature_of_curve 和 torsion_of_curve:
    # 這些通常是針對嵌入在歐幾里得空間的曲線公式 (Frenet-Serret)。
    # 如果你的 Metric 是非平直的 (如球面)，這些公式需要改寫為協變導數形式。
    # 為了保持類別的一致性，建議只有在確認是 "Euclidean Embedding" 時才使用這兩個函式，
    # 或者將其標註為 utility method。

Metric = TensorMetric

# --------------------------------------------------
# Factory Functions (度規工廠函數)
# --------------------------------------------------

def euclidean_metric():
    """
    建立 3D 歐幾里得度規 (直角坐標)。
    坐標: x, y, z
    度規: diag(1, 1, 1)
    """
    x, y, z = sp.symbols('x y z')
    coords = [x, y, z]
    g_matrix = sp.eye(3)
    return TensorMetric(g_matrix, coords)

def spherical_metric():
    """
    建立 3D 球坐標度規。
    坐標: r, theta, phi
    度規: diag(1, r^2, r^2 * sin(theta)^2)
    """
    # 物理坐標通常假設為實數且 r>0
    r, theta, phi = sp.symbols('r theta phi', real=True, positive=True)
    coords = [r, theta, phi]
    g_matrix = sp.diag(1, r**2, r**2 * sp.sin(theta)**2)
    return TensorMetric(g_matrix, coords)

def cylindrical_metric():
    """
    建立 3D 圓柱坐標度規。
    坐標: rho, phi, z
    度規: diag(1, rho^2, 1)
    """
    rho, phi, z = sp.symbols(r'\rho \phi z', real=True, positive=True)
    coords = [rho, phi, z]
    g_matrix = sp.diag(1, rho**2, 1)
    return TensorMetric(g_matrix, coords)

def polar_metric():
    """
    建立 2D 極坐標度規。
    坐標: r, theta
    度規: diag(1, r^2)
    """
    r, theta = sp.symbols('r theta', real=True, positive=True)
    coords = [r, theta]
    g_matrix = sp.diag(1, r**2)
    return TensorMetric(g_matrix, coords)


