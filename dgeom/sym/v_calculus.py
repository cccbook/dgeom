# -------------------------------------------------------------
# 描述：在標準直角坐標系 (歐幾里得空間) 下，使用 SymPy 實作梯度、散度、旋度與線積分。
# -------------------------------------------------------------

import sympy as sp

# --------------------------------------------------
# I. 基礎定義：直角坐標變數
# --------------------------------------------------

# 宣告直角坐標變數
x, y, z = sp.symbols('x y z')
coords = [x, y, z]
dim = 3 # 僅考慮三維空間

# --------------------------------------------------
# II. 向量微積分函式 (標準直角坐標公式)
# --------------------------------------------------

def gradient(f):
    """
    計算純量場 f 的梯度 (Gradient, ∇f)。
    ∇f = [∂f/∂x, ∂f/∂y, ∂f/∂z]
    
    :param f: SymPy 運算式 (純量場)。
    :輸出: 梯度向量 (SymPy Matrix)。
    """
    # 梯度分量即為純量場對各坐標的偏導數
    grad_components = [sp.diff(f, coord) for coord in coords]
    return sp.Matrix(grad_components)


def divergence(F):
    """
    計算向量場 F 的散度 (Divergence, ∇ · F)。
    F = [Fx, Fy, Fz]
    ∇ · F = ∂Fx/∂x + ∂Fy/∂y + ∂Fz/∂z
    
    :param F: SymPy Matrix (向量場的直角分量 [Fx, Fy, Fz])。
    :輸出: 散度純量 (SymPy 運算式)。
    """
    if F.shape[0] != dim:
        raise ValueError("向量場分量數必須為 3 (x, y, z)。")
        
    Fx, Fy, Fz = F[0], F[1], F[2]
    
    # 散度是三個導數的和
    divergence_sum = sp.diff(Fx, x) + sp.diff(Fy, y) + sp.diff(Fz, z)
    
    return sp.simplify(divergence_sum)


def curl(F):
    """
    計算向量場 F 的旋度 (Curl, ∇ × F)。
    F = [Fx, Fy, Fz]
    ∇ × F = [(∂Fz/∂y - ∂Fy/∂z), (∂Fx/∂z - ∂Fz/∂x), (∂Fy/∂x - ∂Fx/∂y)]
    
    :param F: SymPy Matrix (向量場的直角分量 [Fx, Fy, Fz])。
    :輸出: 旋度向量 (SymPy Matrix)。
    """
    if F.shape[0] != dim:
        raise ValueError("旋度運算僅實用於三維空間 (分量數必須為 3)。")
        
    Fx, Fy, Fz = F[0], F[1], F[2]
    
    # X 分量: ∂Fz/∂y - ∂Fy/∂z
    curl_x = sp.diff(Fz, y) - sp.diff(Fy, z)
    
    # Y 分量: ∂Fx/∂z - ∂Fz/∂x
    curl_y = sp.diff(Fx, z) - sp.diff(Fz, x)
    
    # Z 分量: ∂Fy/∂x - ∂Fx/∂y
    curl_z = sp.diff(Fy, x) - sp.diff(Fx, y)
    
    return sp.simplify(sp.Matrix([curl_x, curl_y, curl_z]))


def line_integral(F, path_r, t, ta, tb):
    """
    計算向量場 F 沿著路徑 C 的線積分：∫_C F · dr。
    假設在標準直角坐標系下。
    
    公式: ∫_C F · dr = ∫_{ta}^{tb} F(r(t)) · (dr/dt) dt
    
    :param F: SymPy Matrix (向量場的直角分量 [Fx, Fy, Fz])。
    :param path_r: SymPy Matrix (路徑 r(t) 的坐標 [x(t), y(t), z(t)])。
    :param t: SymPy 符號 (參數化變數)。
    :param ta: 數值或 SymPy 運算式 (積分下限)。
    :param tb: 數值或 SymPy 運算式 (積分上限)。
    :輸出: 線積分的結果 (SymPy 運算式或數值)。
    """
    
    if path_r.shape != F.shape:
        raise ValueError("路徑 r(t) 和向量場 F 的維度必須相同。")
        
    # 1. 計算切線向量 dr/dt
    dr_dt = sp.diff(path_r, t)
    
    # 2. 將路徑 r(t) 代入向量場 F(r(t))
    
    # 確保路徑和坐標變數數量一致
    if len(coords) < path_r.shape[0]:
        raise ValueError("路徑維度超過了定義的 (x, y, z) 坐標數量。")
        
    F_substituted = F
    current_coords = coords[:path_r.shape[0]] # 只使用路徑的維度
    
    for i in range(len(current_coords)):
        # 將向量場 F 中所有的 x 替換為 x(t), y 替換為 y(t), ...
        F_substituted = F_substituted.subs(current_coords[i], path_r[i])
        
    # 3. 計算點積 F(r(t)) · (dr/dt)
    integrand = (F_substituted.T * dr_dt)[0]
    
    # 4. 進行定積分 ∫_{ta}^{tb} (F · dr/dt) dt
    try:
        integral_result = sp.integrate(integrand, (t, ta, tb))
        return sp.simplify(integral_result)
    except NotImplementedError:
        return sp.Integral(integrand, (t, ta, tb))

