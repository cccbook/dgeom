import sympy as sp
import itertools
from functools import reduce
from ._metrics import Metric, get_euclidean_metric
from ._exterior_derivative import *

EUCLIDEAN_METRIC = get_euclidean_metric()

# ==========================================
# Part 2: 區域參數化與積分 (SymPy Version)
# ==========================================

class ParametrizedDomain:
    """
    u_vars: 參數符號列表，如 [u, v]
    u_bounds: 參數範圍，如 [(0, 1), (0, 2*sp.pi)] (數值或符號皆可)
    map_func: 接受 u_vars 符號，回傳物理座標表達式 [x_expr, y_expr, z_expr]
    """
    def __init__(self, u_vars, u_bounds, map_func):
        self.dim = len(u_vars)
        self.u_vars = u_vars
        self.u_bounds = u_bounds
        self.map_func = map_func
        
        # 預計算物理座標表達式
        self.p_exprs = sp.Matrix(map_func(u_vars)) 
        # 假設 map_func 回傳 list 或 Matrix

    def get_tangent_vectors(self, manifold_coords):
        """
        計算參數空間的切向量基底 (Pullback)
        ∂p/∂u_i
        manifold_coords: 用於定義 TangentVector 輸出所在的座標系
        """
        tangents = []
        # Jacobian 矩陣每一行就是一個切向量
        # Jacobian: rows=spatial_dims, cols=param_dims
        J = self.p_exprs.jacobian(self.u_vars)
        
        for i in range(self.dim):
            # 取出第 i 行 (針對第 i 個參數的偏微分)
            vec_components = J.col(i)
            # 必須使用 substitute 將結果中的參數 (u,v) 暫時保留
            # 注意：這裡產生的 TangentVector 內含 u, v 符號，
            # 但它是定義在 manifold_coords (x,y,z) 空間的向量場。
            # 當積分替換變數時，x,y,z 會被換成 u,v 的表達式。
            tangents.append(TangentVector(vec_components, manifold_coords, name=f"d/d{self.u_vars[i]}"))
            
        return tangents

def integrate_form(form, domain, manifold_coords):
    """
    符號積分
    form: k-form
    domain: ParametrizedDomain
    manifold_coords: 定義 form 的空間座標符號 [x, y, z]
    """
    if form.k != domain.dim:
        raise ValueError(f"維度不匹配: Form是{form.k}階, 區域是{domain.dim}維")
    
    # 1. 取得切向量基底 (以參數 u_vars 表示的 x,y,z 分量)
    basis_vectors = domain.get_tangent_vectors(manifold_coords)
    
    # 2. 將 Form 作用於切向量
    # 這會得到一個包含 (x,y,z) 和 (u,v) 混合的表達式，甚至包含導數
    integrand_expr = form(*basis_vectors)
    
    # 3. 變數替換 (Pullback)
    # 將表達式中的 x, y, z 替換為 u, v 的函數
    substitution = dict(zip(manifold_coords, domain.p_exprs))
    integrand_in_uv = integrand_expr.subs(substitution).simplify()
    
    # 4. 執行多重積分
    # 從最後一個變數開始積 (通常順序沒差，除非有相依邊界，這裡假設由 bounds 順序決定)
    result = integrand_in_uv
    
    # 這是為了配合 nquad 的順序習慣，通常是 u1, u2...
    # 但 sympy integrate 是由內而外。
    # 假設 u_vars = [u, v]，bounds = [u_bound, v_bound]
    # 我們依序對 u, 然後對 v 積分
    for param, bounds in zip(domain.u_vars, domain.u_bounds):
        result = sp.integrate(result, (param, bounds[0], bounds[1]))
        
    return result

# ==========================================
# Part 3: 幾何形狀 - 超立方體 (HyperCube)
# ==========================================

class HyperCube(ParametrizedDomain):
    def __init__(self, u_vars, bounds):
        """
        bounds: [(min, max), ...] 數值或符號
        u_vars: 對應的參數符號 [x, y]
        """
        # Identity map
        def identity_map(u):
            return u
        
        # 初始化父類別，這會建立 self.u_bounds
        super().__init__(u_vars, bounds, identity_map)
        self.manifold_coords = u_vars # 自身就是座標系

    def get_boundaries(self):
        boundaries = []
        dim = self.dim
        
        for i in range(dim):
            # 剩餘參數
            sub_vars = self.u_vars[:i] + self.u_vars[i+1:]
            
            # [修正] 使用 self.u_bounds 而非 self.bounds
            sub_bounds = self.u_bounds[:i] + self.u_bounds[i+1:]
            
            # [修正] 使用 self.u_bounds
            val_min = self.u_bounds[i][0]
            val_max = self.u_bounds[i][1]
            
            # 定義 Min 面映射
            # 必須使用 closure 捕捉變數
            def make_map(insert_idx, fixed_val):
                # 注意：這裡要將 list 轉回 tuple 或 list 以便後續處理，
                # 但 Python list 加法會回傳 list，沒問題
                return lambda u_sub: u_sub[:insert_idx] + [fixed_val] + u_sub[insert_idx:]

            map_min = make_map(i, val_min)
            map_max = make_map(i, val_max)
            
            domain_min = ParametrizedDomain(sub_vars, sub_bounds, map_min)
            domain_max = ParametrizedDomain(sub_vars, sub_bounds, map_max)
            
            # 定向: (-1)^i for Max, (-1)^(i+1) for Min
            sign_base = (-1)**i
            boundaries.append((domain_max, sign_base))
            boundaries.append((domain_min, -1 * sign_base))
            
        return boundaries

class ParametricPatch(HyperCube):
    """
    3D 空間中的曲面 Patch
    """
    def __init__(self, u_vars, u_bounds, map_func_3d):
        # 初始化父類別 (參數空間)
        super().__init__(u_vars, u_bounds)
        self.surface_map = map_func_3d # (u,v) -> (x,y,z)
        
        # 覆蓋 map_func 為 3D 映射
        self.map_func = map_func_3d
        self.p_exprs = sp.Matrix(map_func_3d(u_vars))

    def get_boundaries(self):
        # 取得參數空間的邊界 (線段)
        param_boundaries = super().get_boundaries()
        real_boundaries = []
        
        for domain, sign in param_boundaries:
            # 複合映射: t -> uv -> xyz
            # domain.map_func 是 t -> uv
            # self.surface_map 是 uv -> xyz
            
            def composed_map(t_params, local_map=domain.map_func, surf_map=self.surface_map):
                uv = local_map(t_params)
                xyz = surf_map(uv)
                return xyz
            
            real_domain = ParametrizedDomain(domain.u_vars, domain.u_bounds, composed_map)
            real_boundaries.append((real_domain, sign))
            
        return real_boundaries

# --------------------------------------------------
# III. 向量微積分函式 (基於通用張量公式)
# --------------------------------------------------

def d_gradient(f, metric=EUCLIDEAN_METRIC):
    """
    計算純量場 f 的梯度 (Gradient, ∇f)。
    結果為協變向量 (Covariant Vector)，分量為 ∂f/∂x^i。
    ∇f_i = ∂f / ∂x^i
    
    :param f: SymPy 運算式 (純量場)。
    :param metric: Metric 實例。
    :輸出: 梯度向量的協變分量 (SymPy Matrix)。
    """
    # 梯度分量即為純量場對各坐標的偏導數
    grad_components = [sp.diff(f, coord) for coord in metric.coords]
    return sp.Matrix(grad_components)


def d_divergence(F_contravariant, metric=EUCLIDEAN_METRIC):
    """
    計算向量場 F 的散度 (Divergence, ∇ · F)。
    公式使用逆變分量 F^i：
    ∇ · F = (1/√|g|) * Σ ∂/∂x^i (√|g| * F^i) 
    
    :param F_contravariant: SymPy Matrix (向量場的逆變分量 [F^1, F^2, F^3])。
    :param metric: Metric 實例。
    :輸出: 散度純量 (SymPy 運算式)。
    """
    if F_contravariant.shape[0] != metric.dim:
        raise ValueError("向量場分量數與度規維度不匹配。")
        
    coords = metric.coords
    sqrt_det_g = metric.sqrt_det_g
    
    divergence_sum = 0
    for i in range(metric.dim):
        # 內層項: √|g| * F^i
        inner_term = sqrt_det_g * F_contravariant[i]
        # 偏導數: ∂/∂x^i (√|g| * F^i)
        derivative = sp.diff(inner_term, coords[i])
        divergence_sum += derivative
    
    # 最終結果: (1/√|g|) * Σ derivative
    return sp.simplify((1 / sqrt_det_g) * divergence_sum)

def d_curl(F_covariant, metric=EUCLIDEAN_METRIC):
    """
    計算向量場 F 的旋度 (Curl, ∇ × F)。(僅適用於 3D)
    
    首先計算旋度向量的逆變分量 (Curl F)^k，然後利用度規 g_ij 轉為協變分量 (Curl F)_i。
    
    :param F_covariant: SymPy Matrix (向量場的協變分量 [F_1, F_2, F_3])。
    :param metric: Metric 實例。
    :輸出: 旋度向量的協變分量 (SymPy Matrix)。
    """
    if metric.dim != 3:
        raise ValueError("旋度運算僅實用於三維空間。")
        
    coords = metric.coords
    sqrt_det_g = metric.sqrt_det_g
    F1, F2, F3 = F_covariant[0], F_covariant[1], F_covariant[2]
    
    # --------------------------------------------------
    # 計算旋度向量的逆變分量 (Curl F)^k
    # 公式: (∇ × F)^k = (1/√|g|) * ε^ijk * (∂F_j / ∂x^i)
    # --------------------------------------------------
    
    # 1. (∇ × F)^1: (1/√|g|) * (∂F_3/∂x^2 - ∂F_2/∂x^3)
    curl_contravariant_1 = (1 / sqrt_det_g) * (sp.diff(F3, coords[1]) - sp.diff(F2, coords[2]))
    
    # 2. (∇ × F)^2: (1/√|g|) * (∂F_1/∂x^3 - ∂F_3/∂x^1)
    curl_contravariant_2 = (1 / sqrt_det_g) * (sp.diff(F1, coords[2]) - sp.diff(F3, coords[0]))
    
    # 3. (∇ × F)^3: (1/√|g|) * (∂F_2/∂x^1 - ∂F_1/∂x^2)
    curl_contravariant_3 = (1 / sqrt_det_g) * (sp.diff(F2, coords[0]) - sp.diff(F1, coords[1]))
    
    curl_contravariant = sp.Matrix([curl_contravariant_1, curl_contravariant_2, curl_contravariant_3])
    
    # 將逆變分量轉換為協變分量 (Curl F)_i = g_ij * (Curl F)^j (降指標)
    curl_covariant = metric.g * curl_contravariant
    
    return sp.simplify(curl_covariant)

# --------------------------------------------------
# IV. 線積分函式 (Line Integral)
# --------------------------------------------------

def d_line_integral(F_contravariant, path_r, t, ta, tb, metric=EUCLIDEAN_METRIC):
    """
    計算向量場 F 沿著路徑 C 的線積分：∫_C F · dr。
    此函式假設在歐幾里得空間 (直角坐標系) 下進行，
    因此協變和逆變分量是相同的，且基底是正交的。
    
    公式: ∫_C F · dr = ∫_{ta}^{tb} F(r(t)) · (dr/dt) dt
    
    :param F_contravariant: SymPy Matrix (向量場的逆變/直角分量 [F^1, F^2, F^3])。
    :param path_r: SymPy Matrix (路徑 r(t) 的坐標 [x(t), y(t), z(t)])。
    :param t: SymPy 符號 (參數化變數)。
    :param ta: 數值或 SymPy 運算式 (積分下限)。
    :param tb: 數值或 SymPy 運算式 (積分上限)。
    :param metric: Metric 實例 (默認為歐幾里得度規)。
    :輸出: 線積分的結果 (SymPy 運算式或數值)。
    """
    
    if path_r.shape != F_contravariant.shape:
        raise ValueError("路徑 r(t) 和向量場 F 的維度必須相同。")
    if metric is not EUCLIDEAN_METRIC:
        # 警告：此簡單實現僅適用於歐幾里得空間，非歐幾里得空間需要更複雜的度規操作。
        print("警告：線積分的簡單實現假設為歐幾里得空間。")

    # 1. 計算切線向量 dr/dt
    dr_dt = sp.diff(path_r, t)
    
    # 2. 將路徑 r(t) 代入向量場 F(r(t))
    # 這裡假設坐標變數就是 metric.coords (例如：x, y, z)
    coords = metric.coords
    
    # 確保路徑和坐標變數數量一致
    if len(coords) != path_r.shape[0]:
        raise ValueError("路徑維度與坐標變數數量不匹配。")
        
    F_substituted = F_contravariant
    for i in range(len(coords)):
        # 將向量場 F 中所有的 x 替換為 x(t), y 替換為 y(t), ...
        F_substituted = F_substituted.subs(coords[i], path_r[i])
        
    # 3. 計算點積 F(r(t)) · (dr/dt)
    # 點積 F · (dr/dt) = F_x * (dx/dt) + F_y * (dy/dt) + ...
    integrand = (F_substituted.T * dr_dt)[0] # SymPy 矩陣乘法，[0] 取出純量結果
    
    # 4. 進行定積分 ∫_{ta}^{tb} (F · dr/dt) dt
    try:
        integral_result = sp.integrate(integrand, (t, ta, tb))
        return sp.simplify(integral_result)
    except NotImplementedError:
        print("警告: SymPy 無法解析此積分，返回未計算的積分運算式。")
        return sp.Integral(integrand, (t, ta, tb))
