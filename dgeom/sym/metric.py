import sympy as sp
import numpy as np
from sympy import MutableDenseNDimArray, Matrix, diff
from .tensor import GeometricTensor

class MetricTensor(GeometricTensor):
    """
    度規張量 (self Tensor)。
    
    數學定義：
    1. Rank 2 協變張量 (Covariant)
    2. 對稱 (Symmetric)
    3. 非退化 (Non-degenerate / Invertible)
    
    功能：
    提供幾何測量 (內積、長度、角度) 與 幾何結構 (聯絡、曲率)。
    """

    def __init__(self, data, coords):
        # 1. 基礎張量初始化 (強制為協變 Rank 2)
        super().__init__(data, coords, [-1, -1])
        
        # 2. 數學性質檢查
        self._validate_metric_properties()

    def _validate_metric_properties(self):
        """驗證度規的數學性質"""
        rows, cols = self.data.shape
        
        # A. 方陣檢查
        if rows != cols:
            raise ValueError("度規張量必須是方陣 (Square Matrix)。")
            
        # 轉為 Matrix 以便進行代數檢查
        g_mat = Matrix(self.data.tolist())
        
        # B. 對稱性檢查 (g_ij = g_ji)
        # 注意: 符號運算有時無法自動化簡為 0，所以我們只給警告，不強制報錯
        if not g_mat.is_symmetric():
            # 嘗試化簡後再檢查一次
            simplified_mat = sp.simplify(g_mat)
            if not simplified_mat.is_symmetric():
                print("Warning: 輸入的度規張量似乎不對稱，這可能違反黎曼幾何定義。")

        # C. 非退化檢查 (Determinant != 0)
        # 我們將其存為屬性，因為後續計算常會用到行列式
        self.determinant = sp.simplify(g_mat.det())
        if self.determinant == 0:
            raise ValueError("度規張量是退化的 (Singular)，行列式為 0，無法定義逆度規。")

    # ==========================================
    # 幾何測量 (Measurements) - 新增功能
    # ==========================================

    def inner_product(self, u, v):
        """
        計算兩個向量的內積 <u, v> = g_ij * u^i * v^j
        
        Args:
            u, v: GeometricTensor (必須是逆變向量 Rank 1, [1])
                  或是 list/Matrix (視為逆變分量)
        Returns:
            SymPy Expression (Scalar)
        """
        # 確保輸入是張量物件
        u = self._ensure_vector(u)
        v = self._ensure_vector(v)
        
        # 檢查是否為逆變向量 (Contravariant)
        if u.index_config != [1] or v.index_config != [1]:
            raise ValueError("內積必須作用於兩個逆變向量 (Contravariant Vectors) 上。")

        # 計算: sum(g_ij * u^i * v^j)
        # 這裡直接操作 data 以獲得最佳效能，避免建立過多中間張量物件
        res = 0
        dim = self.dim
        for i in range(dim):
            for j in range(dim):
                res += self.data[i, j] * u.data[i] * v.data[j]
                
        return sp.simplify(res)

    def norm(self, v):
        """
        計算向量的長度 (Norm) ||v|| = sqrt(<v, v>)
        注意：在相對論中，類光向量長度為 0，類空/類時向量長度平方可能為負。
        """
        v_squared = self.inner_product(v, v)
        return sp.sqrt(v_squared)

    def angle(self, u, v):
        """
        計算兩個向量的夾角 (Angle)
        cos(theta) = <u, v> / (||u|| * ||v||)
        """
        prod = self.inner_product(u, v)
        len_u = self.norm(u)
        len_v = self.norm(v)
        
        # 檢查分母是否為 0 (如光速向量)
        if len_u == 0 or len_v == 0:
            raise ValueError("無法計算零向量或類光向量的夾角。")
            
        return sp.acos(prod / (len_u * len_v))

    def _ensure_vector(self, v):
        """輔助函式：將輸入轉為逆變張量"""
        if isinstance(v, GeometricTensor):
            return v
        # 假設 raw input 是逆變分量 list
        return GeometricTensor(v, self.coords, [1])

    # ==========================================
    # 幾何結構 (Geometric Structures)
    # ==========================================

    def inverse(self):
        """計算逆度規 g^uv"""
        g_mat = Matrix(self.data.tolist())
        # 因為我們在 init 檢查過行列式，這裡 inv() 應該是安全的
        g_inv_mat = g_mat.inv()
        return GeometricTensor(MutableDenseNDimArray(g_inv_mat.tolist()), self.coords, [1, 1])

    def christoffel_symbols(self):
        """計算 Christoffel Gamma^k_ij"""
        dim = self.dim
        coords = self.coords
        g_inv = self.inverse()
        
        # 偏導數預計算 partial_g[k][i][j] = d_k g_ij
        partial_g = [[[diff(self.data[i, j], x_k) for j in range(dim)] 
                      for i in range(dim)] 
                     for x_k in coords]

        gamma_data = sp.MutableDenseNDimArray.zeros(dim, dim, dim)

        for k in range(dim):
            for i in range(dim):
                for j in range(dim):
                    val = 0
                    for l in range(dim):
                        # Gamma公式
                        term = partial_g[j][i][l] + partial_g[i][j][l] - partial_g[l][i][j]
                        val += g_inv.data[k, l] * term
                    gamma_data[k, i, j] = sp.simplify(0.5 * val)
                    
        return GeometricTensor(gamma_data, coords, [1, -1, -1])

    def riemann_tensor(self):
        """計算 Riemann R^rho_sigma,mu,nu"""
        gamma = self.christoffel_symbols()
        G = gamma.data
        dim = self.dim
        coords = self.coords
        
        R_data = sp.MutableDenseNDimArray.zeros(dim, dim, dim, dim)
        
        for rho in range(dim):
            for sigma in range(dim):
                for mu in range(dim):
                    for nu in range(dim):
                        t1 = diff(G[rho, nu, sigma], coords[mu])
                        t2 = diff(G[rho, mu, sigma], coords[nu])
                        t3 = 0
                        t4 = 0
                        for lam in range(dim):
                            t3 += G[rho, mu, lam] * G[lam, nu, sigma]
                            t4 += G[rho, nu, lam] * G[lam, mu, sigma]
                        R_data[rho, sigma, mu, nu] = sp.simplify(t1 - t2 + t3 - t4)
                        
        return GeometricTensor(R_data, coords, [1, -1, -1, -1])

    def ricci_tensor(self):
        """計算 Ricci R_mu,nu"""
        riemann = self.riemann_tensor()
        return riemann.contract(0, 2)

    def ricci_scalar(self):
        """計算 Ricci Scalar R"""
        ricci = self.ricci_tensor()
        g_inv = self.inverse()
        mixed = g_inv.tensor_product(ricci)
        return sp.simplify(mixed.contract(0, 2).contract(0, 1).data[()])

    # ... (保留 arc_length, solve_geodesic_bvp 等方法) ...
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
        """生成測地線方程"""
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
        """數值求解測地線"""
        try:
            from scipy.integrate import solve_bvp
        except ImportError:
            raise ImportError("需安裝 scipy")
        Gamma = self.christoffel_symbols()
        gamma_func = sp.lambdify(self.coords, Gamma.data.tolist(), modules='numpy')
        dim = self.dim
        def ode_system(t, y):
            pos = y[:dim]; vel = y[dim:]; acc = np.zeros_like(vel)
            for i in range(pos.shape[1]):
                p = pos[:, i]; v = vel[:, i]
                G = np.array(gamma_func(*p))
                acc[:, i] = -np.einsum('kij,i,j->k', G, v, v)
            return np.vstack((vel, acc))
        def bc(ya, yb):
            return np.concatenate((ya[:dim] - start_point, yb[:dim] - end_point))
        x = np.linspace(0, 1, num_points)
        y_guess = np.zeros((2*dim, num_points))
        for i in range(dim):
            y_guess[i] = np.linspace(start_point[i], end_point[i], num_points)
            y_guess[dim+i] = end_point[i] - start_point[i]
        res = solve_bvp(ode_system, bc, x, y_guess, tol=1e-3)
        return res.y[:dim]

    def _apply_index_map(self, tensor, pos, new_type):
        """
        對給定的張量物件執行指標升降操作。
        """
        
        if not isinstance(tensor, GeometricTensor):
            raise TypeError("輸入必須是 GeometricTensor 實例。")
            
        rank = tensor.rank
        if pos < 0 or pos >= rank:
            raise IndexError("指標位置超出範圍。")
            
        current_type = tensor.index_config[pos]
        
        if current_type == new_type:
            # print(f"指標位置 {pos} 已是類型 {new_type}，無需操作。")
            return tensor.tensor_product(GeometricTensor([], tensor.coords, [])) 
        
        # 選擇度規
        if current_type == -1 and new_type == 1:
            Metric = self.inverse() # g^ij
        elif current_type == 1 and new_type == -1:
            Metric = self # g_ij
        else:
            raise ValueError("new_type 必須是 1 (升) 或 -1 (降)")
            
        # 1. 張量積：索引順序變為 (Metric_1, Metric_2, Tensor_1, Tensor_2, ...)
        product = Metric.tensor_product(tensor)
        
        # 2. 縮併：Metric 的第 2 個指標 (idx 1) 與 Tensor 的目標指標 (idx pos+2)
        contracted = product.contract(1, pos + 2)
        
        # 3. 指標重排 (Permute)：
        # 縮併後的 array 索引結構：
        #   Index 0: 來自 Metric 的新指標
        #   Index 1 ~ (rank-1): 來自 Tensor 的其餘指標 (保持原本相對順序)
        
        # 我們需要建立一個 permutation list，告訴 permutedims 新的軸要取自舊的哪個軸
        # 目標：
        #   將舊軸 0 (Metric index) 放到位置 pos
        #   將舊軸 1 ~ rank-1 依序填入剩餘位置
        
        # 步驟 A: 取出除了 Metric index 以外的所有舊軸索引 (即 1 到 rank-1)
        perm_list = list(range(1, rank))
        
        # 步驟 B: 將 Metric index (0) 插入到目標位置 pos
        perm_list.insert(pos, 0)
        
        # 執行重排
        final_data = sp.permutedims(contracted.data, perm_list)

        # 4. 更新指標配置
        new_config = list(tensor.index_config)
        new_config[pos] = new_type
        
        return GeometricTensor(final_data, tensor.coords, new_config)

    def raise_index(self, tensor, pos):
        """將張量在 pos 位置的協變指標 (下標) 升為逆變指標 (上標)"""
        if tensor.index_config[pos] == 1:
            return tensor # 已經是上標
        return self._apply_index_map(tensor, pos, new_type=1)

    def lower_index(self, tensor, pos):
        """將張量在 pos 位置的逆變指標 (上標) 降為協變指標 (下標)"""
        if tensor.index_config[pos] == -1:
            return tensor # 已經是下標
        return self._apply_index_map(tensor, pos, new_type=-1)

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
