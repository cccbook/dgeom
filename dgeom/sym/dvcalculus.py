# vcalculus.py (最終完整版 - 通用張量度規與歐幾里得預設)
# -------------------------------------------------------------
# 描述：在一般曲線坐標系下，使用 SymPy 實作梯度、散度與旋度運算。
# 函式使用通用張量公式，並預設為歐幾里得度規 (直角坐標系)。
# -------------------------------------------------------------

import sympy as sp
from dgeom.sym.metrics import Metric, EUCLIDEAN_METRIC

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

