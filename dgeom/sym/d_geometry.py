import sympy as sp
import itertools
from sympy import MutableDenseNDimArray, Matrix, diff
from .metric import MetricTensor, GeometricTensor, euclidean_metric
from .d_operator import TangentVector, Form 

# 預設歐幾里得度規 (新版)
EUCLIDEAN_METRIC = euclidean_metric()

# ==========================================
# Part 1: 區域參數化與積分 (Integration)
# ==========================================

class ParametrizedDomain:
    """
    參數化區域 (Parametrized Domain)
    將參數空間 u_vars 映射到物理空間 manifold_coords
    
    Attributes:
        u_vars: 參數符號列表 [u, v]
        u_bounds: 參數範圍 [(0, 1), (0, 2*pi)]
        map_func: 映射函數 (u, v) -> [x, y, z]
    """
    def __init__(self, u_vars, u_bounds, map_func):
        self.dim = len(u_vars)
        self.u_vars = u_vars
        self.u_bounds = u_bounds
        self.map_func = map_func
        
        # 預計算物理座標表達式
        # 這裡仍維持 sp.Matrix 以利用方便的 jacobian 方法
        self.p_exprs = sp.Matrix(map_func(u_vars)) 

    def get_tangent_vectors(self, manifold_coords):
        """
        計算參數空間的切向量基底 (Pullback / Jacobian columns)
        回傳: List[TangentVector]
        """
        tangents = []
        # Jacobian: rows=spatial_dims, cols=param_dims
        J = self.p_exprs.jacobian(self.u_vars)
        
        for i in range(self.dim):
            # 取出第 i 行 (針對第 i 個參數的偏微分)
            # 轉換為 list 格式
            vec_components = [J[row, i] for row in range(J.rows)]
            
            # 使用新版 TangentVector (繼承 GeometricTensor)
            name = f"∂/∂{self.u_vars[i]}"
            tangents.append(TangentVector(vec_components, manifold_coords, name=name))
            
        return tangents

def integrate_form(form, domain, manifold_coords):
    """
    對微分形式進行符號積分 (Integration of Differential Forms)
    Integral_D omega = Integral_U omega(Tp_1, ..., Tp_k) du
    """
    if form.k != domain.dim:
        raise ValueError(f"維度不匹配: Form是{form.k}階, 區域是{domain.dim}維")
    
    # 1. 取得切向量基底 (以參數 u_vars 表示的 x,y,z 分量)
    basis_vectors = domain.get_tangent_vectors(manifold_coords)
    
    # 2. 將 Form 作用於切向量 (Pullback)
    # Form 會自動處理代入與行列式計算
    integrand_expr = form(*basis_vectors)
    
    # 3. 變數替換
    # 將表達式中的 x, y, z 替換為 u, v 的函數
    substitution = dict(zip(manifold_coords, domain.p_exprs))
    integrand_in_uv = integrand_expr.subs(substitution).simplify()
    
    # 4. 執行多重積分
    result = integrand_in_uv
    
    # 依序對參數積分
    for param, bounds in zip(domain.u_vars, domain.u_bounds):
        result = sp.integrate(result, (param, bounds[0], bounds[1]))
        
    return result

# ==========================================
# Part 2: 幾何形狀 (Shapes)
# ==========================================

class HyperCube(ParametrizedDomain):
    """超立方體 (參數即座標)"""
    def __init__(self, u_vars, bounds):
        # Identity map: u -> u
        super().__init__(u_vars, bounds, lambda u: u)
        self.manifold_coords = u_vars 

    def get_boundaries(self):
        """取得邊界 (遞迴定義)"""
        boundaries = []
        dim = self.dim
        
        for i in range(dim):
            # 剩餘參數
            sub_vars = self.u_vars[:i] + self.u_vars[i+1:]
            sub_bounds = self.u_bounds[:i] + self.u_bounds[i+1:]
            
            val_min = self.u_bounds[i][0]
            val_max = self.u_bounds[i][1]
            
            # Closure map factory
            def make_map(insert_idx, fixed_val):
                return lambda u_sub: u_sub[:insert_idx] + [fixed_val] + u_sub[insert_idx:]

            map_min = make_map(i, val_min)
            map_max = make_map(i, val_max)
            
            domain_min = ParametrizedDomain(sub_vars, sub_bounds, map_min)
            domain_max = ParametrizedDomain(sub_vars, sub_bounds, map_max)
            
            # 定向 (Orientation)
            sign_base = (-1)**i
            boundaries.append((domain_max, sign_base))
            boundaries.append((domain_min, -1 * sign_base))
            
        return boundaries

class ParametricPatch(HyperCube):
    """3D 空間中的曲面 Patch"""
    def __init__(self, u_vars, u_bounds, map_func_3d):
        super().__init__(u_vars, u_bounds)
        self.surface_map = map_func_3d
        
        # 覆蓋映射邏輯
        self.map_func = map_func_3d
        self.p_exprs = sp.Matrix(map_func_3d(u_vars))

    def get_boundaries(self):
        param_boundaries = super().get_boundaries()
        real_boundaries = []
        
        for domain, sign in param_boundaries:
            # 複合映射: t -> uv -> xyz
            def composed_map(t_params, local_map=domain.map_func, surf_map=self.surface_map):
                uv = local_map(t_params)
                return surf_map(uv)
            
            real_domain = ParametrizedDomain(domain.u_vars, domain.u_bounds, composed_map)
            real_boundaries.append((real_domain, sign))
            
        return real_boundaries

# ==========================================
# Part 3: 向量微積分 (MetricTensor Version)
# ==========================================

def d_gradient(f, metric):
    """
    計算梯度 (Gradient ∇f)。
    回傳: Rank 1 協變張量 (Covariant Vector) [-1]
    分量: ∂f/∂x^i
    """
    if not isinstance(metric, MetricTensor):
        raise TypeError("Metric 必須是 MetricTensor 物件")
        
    coords = metric.coords
    # 計算偏導數
    grad_comps = [sp.diff(f, x) for x in coords]
    
    return GeometricTensor(grad_comps, coords, [-1])

def d_divergence(F, metric):
    """
    計算散度 (Divergence ∇·F)。
    公式: ∇·F = (1/√|g|) * ∂_i (√|g| * F^i)
    
    Args:
        F: 向量場 (GeometricTensor, Matrix, 或 list)
           若為 GeometricTensor 且為協變 ([-1])，會自動升指標。
    """
    # 1. 統一轉為 GeometricTensor (逆變形式 [1])
    if hasattr(F, 'data'):
        F_tensor = F
    else:
        # 假設 raw list/matrix 是逆變分量
        F_tensor = GeometricTensor(F, metric.coords, [1])
        
    # 如果是協變向量 ([-1])，升指標轉為逆變 ([1])
    if F_tensor.index_config == [-1]:
        g_inv = metric.inverse() # [1, 1]
        # g^ij F_j -> F^i
        # Product: [1, 1, -1], Contract index 1 & 2
        prod = g_inv.tensor_product(F_tensor)
        F_tensor = prod.contract(1, 2)
    elif F_tensor.index_config != [1]:
        raise ValueError("散度運算需要 Rank 1 向量")

    coords = metric.coords
    dim = metric.dim
    
    # 2. 計算行列式 det(g)
    # MetricTensor.data 是 NDimArray，轉為 Matrix 計算行列式較方便
    g_mat = Matrix(metric.data.tolist())
    det_g = g_mat.det()
    sqrt_g = sp.sqrt(sp.Abs(det_g))
    
    # 3. 散度公式求和
    div_sum = 0
    for i in range(dim):
        F_i = F_tensor.data[i] # 這是 F^i
        term = sqrt_g * F_i
        div_sum += sp.diff(term, coords[i])
        
    return sp.simplify((1 / sqrt_g) * div_sum)

def d_curl(F, metric):
    """
    計算旋度 (Curl ∇×F)。僅適用於 3D。
    回傳: Rank 1 協變張量 (Covariant Vector) [-1]
    
    公式: (Curl F)^k = (1/√|g|) * ε^ijk * ∂_i F_j
    最後利用 metric 降指標回傳 F_m。
    """
    if metric.dim != 3:
        raise ValueError("旋度運算僅實用於三維空間。")
        
    # 1. 統一轉為 GeometricTensor (協變形式 [-1])
    # 旋度本質是外微分 d(1-form)，所以輸入應為 1-form (協變)
    if hasattr(F, 'data'):
        F_tensor = F
    else:
        F_tensor = GeometricTensor(F, metric.coords, [-1])
        
    # 如果是逆變向量 ([1])，降指標轉為協變 ([-1])
    if F_tensor.index_config == [1]:
        # g_ij F^j -> F_i
        prod = metric.tensor_product(F_tensor) # [-1, -1, 1]
        F_tensor = prod.contract(1, 2)
    elif F_tensor.index_config != [-1]:
        raise ValueError("旋度運算需要 Rank 1 向量")

    coords = metric.coords
    F_data = F_tensor.data # F_j
    
    # 2. 計算 sqrt|g|
    g_mat = Matrix(metric.data.tolist())
    sqrt_g = sp.sqrt(sp.Abs(g_mat.det()))
    
    # 3. 計算逆變旋度分量 (Curl F)^k using Levi-Civita
    # (Curl F)^1 = (∂2 F3 - ∂3 F2) / sqrt_g
    # 索引對應: 1->0, 2->1, 3->2
    
    # term1: dF_z/dy - dF_y/dz
    c1 = (sp.diff(F_data[2], coords[1]) - sp.diff(F_data[1], coords[2])) / sqrt_g
    # term2: dF_x/dz - dF_z/dx
    c2 = (sp.diff(F_data[0], coords[2]) - sp.diff(F_data[2], coords[0])) / sqrt_g
    # term3: dF_y/dx - dF_x/dy
    c3 = (sp.diff(F_data[1], coords[0]) - sp.diff(F_data[0], coords[1])) / sqrt_g
    
    curl_contra = GeometricTensor([c1, c2, c3], coords, [1])
    
    # 4. 降指標轉為協變分量 (物理上通常將 curl 結果視為向量場，為了一致性轉回協變)
    # Curl_cov = g * Curl_contra
    prod = metric.tensor_product(curl_contra) # [-1, -1, 1]
    curl_cov = prod.contract(1, 2) # [-1]
    
    return curl_cov

# ==========================================
# Part 4: 線積分 (Line Integral)
# ==========================================

def d_line_integral(F, path_r, t, ta, tb, metric=None):
    """
    計算線積分 ∫_C F · dr = ∫ (g_ij F^i dx^j/dt) dt
    
    Args:
        F: 向量場 (GeometricTensor [1] 或 [1] 的 list/Matrix)
        path_r: 路徑參數式 [x(t), y(t), z(t)]
        t: 參數符號
        ta, tb: 積分範圍
        metric: MetricTensor (若為 None 則假設歐幾里得)
    """
    # 1. 處理度規與坐標
    if metric is None:
        # Fallback to Euclidean behavior if metric not provided
        # 嘗試從 F 推斷坐標，或拋出錯誤。這裡假設 F 若不是 Tensor 則使用 path 變數
        # 建議: 強制要求 metric
        metric = EUCLIDEAN_METRIC
        # 檢查維度是否匹配，若不匹配需警告 (此處簡化處理)
    
    coords = metric.coords
    dim = metric.dim
    
    # 2. 統一 F 為 list 格式 (逆變分量)
    if hasattr(F, 'data'):
        # 若傳入協變，需升指標嗎？
        # 線積分通常定義為 <F, dr>，如果是 1-form (協變) 則直接縮併 dr
        # 如果是 Vector (逆變)，則需透過 g 內積
        # 這裡假設輸入 F 代表「力場」或「物理向量」，即逆變分量 F^i
        # 為了計算內積 g_ij F^i v^j
        if F.index_config == [-1]:
            # 視為 1-form F_i，則積分為 F_i * (dx^i/dt)
            # 這不需要度規 g，直接縮併即可
            use_metric_inner_product = False
            F_comps = F.data.tolist()
        else:
            # 視為向量 F^i，積分為 g_ij F^i (dx^j/dt)
            use_metric_inner_product = True
            F_comps = F.data.tolist()
    else:
        # Raw list assumption: Contravariant Vector F^i
        use_metric_inner_product = True
        F_comps = list(F)
        if hasattr(F, 'tolist'): F_comps = F.tolist() # Matrix handling
        if isinstance(F_comps[0], list): F_comps = [c[0] for c in F_comps] # Flatten

    # 3. 計算 dr/dt
    dr_dt = [sp.diff(expr, t) for expr in path_r]
    
    # 4. 變數替換: 將 F(x) 中的坐標換成 x(t)
    subs_map = dict(zip(coords, path_r))
    F_t = [sp.sympify(c).subs(subs_map) for c in F_comps]
    
    # 5. 組裝被積函數
    integrand = 0
    
    if not use_metric_inner_product:
        # F 是 1-form (協變): Sum (F_i * dx^i/dt)
        for fi, dri in zip(F_t, dr_dt):
            integrand += fi * dri
    else:
        # F 是 Vector (逆變): Sum (g_ij * F^i * dx^j/dt)
        g_data = metric.data.tolist()
        
        # 度規也可能隨位置變化 (如球坐標)，需替換
        g_t = [[sp.sympify(val).subs(subs_map) for val in row] for row in g_data]
        
        for i in range(dim):
            for j in range(dim):
                integrand += g_t[i][j] * F_t[i] * dr_dt[j]
                
    # 6. 積分
    try:
        return sp.integrate(integrand, (t, ta, tb))
    except NotImplementedError:
        return sp.Integral(integrand, (t, ta, tb))