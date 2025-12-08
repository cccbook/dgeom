# vcalculus.py (版本 4.0 - 通用張量度規)
# -------------------------------------------------------------
# 描述：在一般曲線坐標系下，使用 SymPy 實作梯度、散度與旋度運算。
# 函式使用通用張量公式，並預設為歐幾里得度規。
# -------------------------------------------------------------

import sympy as sp

# --------------------------------------------------
# 基礎定義：直角坐標系 (歐幾里得度規)
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

# --------------------------------------------------
# Metric 類別 (度規數據封裝)
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
        # 使用 sp.Abs(self.det_g) 確保行列式為非負數 (SymPy 的 sp.sqrt 處理)
        self.sqrt_det_g = sp.sqrt(sp.Abs(self.det_g))

# 建立預設的歐幾里得度規實例
EUCLIDEAN_METRIC = Metric(euclidean_g_matrix, euclidean_coords)

# --------------------------------------------------
# 向量微積分函式 (基於通用張量公式)
# --------------------------------------------------

def gradient(f, metric=EUCLIDEAN_METRIC):
    """
    計算純量場 f 的梯度 (Gradient, ∇f)。
    結果為協變向量 (Covariant Vector)，分量為 ∂f/∂x^i。
    ∇f_i = ∂f / ∂x^i
    
    :param f: SymPy 運算式 (純量場)。
    :param metric: Metric 實例。
    :輸出: 梯度向量的協變分量 (SymPy Matrix)。
    """
    grad_components = [sp.diff(f, coord) for coord in metric.coords]
    return sp.Matrix(grad_components)


def divergence(F_contravariant, metric=EUCLIDEAN_METRIC):
    """
    計算向量場 F 的散度 (Divergence, ∇ · F)。
    公式使用逆變分量 F^i：
    ∇ · F = (1/√|g|) * Σ ∂/∂x^i (√|g| * F^i) (愛因斯坦約定下，Σ 是隱含的)
    
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


def curl(F_covariant, metric=EUCLIDEAN_METRIC):
    """
    計算向量場 F 的旋度 (Curl, ∇ × F)。(僅適用於 3D)
    旋度的結果是一個協變張量 (rank 2)，但在 3D 中，我們通常將其視為一個向量場。
    旋度向量的協變分量 (Curl F)_i 可由下式得出 (使用列維-奇維塔符號):
    (∇ × F)^k = (1/√|g|) * ε^ijk * (∂F_j / ∂x^i)
    
    為了輸出一個 3D 向量，我們計算結果向量的逆變分量 (Curl F)^k，然後再用度規將其轉為協變分量。
    
    簡化公式 (結果為協變分量):
    (∇ × F)_i = (g_ik g_jl - g_il g_jk) * ...

    ***採用更常見的協變向量 $A_i$ (分量 $A_1, A_2, A_3$) 的公式***
    結果為逆變向量 (Curl F)^i 的分量：
    (∇ × F)^k = (1/√|g|) * ε^ijk * (∂F_j / ∂x^i) 
    
    :param F_covariant: SymPy Matrix (向量場的協變分量 [F_1, F_2, F_3])。
    :param metric: Metric 實例。
    :輸出: 旋度向量的協變分量 (SymPy Matrix)。
    """
    if metric.dim != 3:
        raise ValueError("旋度運算僅實用於三維空間。")
        
    coords = metric.coords
    sqrt_det_g = metric.sqrt_det_g
    F1, F2, F3 = F_covariant[0], F_covariant[1], F_covariant[2]
    
    # 計算旋度向量的逆變分量 (Curl F)^k
    # (∇ × F)^1, (∇ × F)^2, (∇ × F)^3
    
    # 1. (∇ × F)^1: (1/√|g|) * (∂F_3/∂x^2 - ∂F_2/∂x^3)
    curl_contravariant_1 = (1 / sqrt_det_g) * (sp.diff(F3, coords[1]) - sp.diff(F2, coords[2]))
    
    # 2. (∇ × F)^2: (1/√|g|) * (∂F_1/∂x^3 - ∂F_3/∂x^1)
    curl_contravariant_2 = (1 / sqrt_det_g) * (sp.diff(F1, coords[2]) - sp.diff(F3, coords[0]))
    
    # 3. (∇ × F)^3: (1/√|g|) * (∂F_2/∂x^1 - ∂F_1/∂x^2)
    curl_contravariant_3 = (1 / sqrt_det_g) * (sp.diff(F2, coords[0]) - sp.diff(F1, coords[1]))
    
    curl_contravariant = sp.Matrix([curl_contravariant_1, curl_contravariant_2, curl_contravariant_3])
    
    # 將逆變分量轉換為協變分量 (Curl F)_i = g_ij * (Curl F)^j
    curl_covariant = metric.g * curl_contravariant
    
    return sp.simplify(curl_covariant)


# --------------------------------------------------
# 測試範例
# --------------------------------------------------

if __name__ == "__main__":
    print("🚀 向量微積分 (vcalculus) 模組測試：通用張量版本")
    print("--------------------------------------------------")

    # --- 1. 歐幾里得度規 (直角坐標 x, y, z) 測試 ---
    print("## 範例 1: 歐幾里得度規 (直角坐標) - 預設參數 (協變=逆變)")
    
    # 純量場 f = x*y*z
    f_euc = x * y * z
    grad_f_euc = gradient(f_euc) 
    print(f"f = {f_euc}")
    print(f"梯度 ∇f (協變分量): {grad_f_euc}")
    # 預期結果: [y*z, x*z, x*y] (∂f/∂x, ∂f/∂y, ∂f/∂z)

    # 向量場 F (協變/逆變分量相同): F^i = [x, y, 0]
    F_euc = sp.Matrix([x, y, 0])
    div_F_euc = divergence(F_euc) # 傳入逆變分量
    curl_F_euc = curl(F_euc)       # 傳入協變分量
    print(f"F (逆變/協變) = {F_euc}")
    print(f"散度 ∇·F: {div_F_euc}")
    # 預期結果: ∂x/∂x + ∂y/∂y + ∂0/∂z = 1 + 1 + 0 = 2
    print(f"旋度 ∇×F (協變分量): {curl_F_euc}")
    # 預期結果: [0, 0, 0]

    # --------------------------------------------------
    print("\n" + "---" * 15 + "\n")
    
    # --- 2. 圓柱坐標系 (非正交，非預設度規) 測試 ---
    # 圓柱坐標 (r, phi, z)，使用 r1, r2, r3 表示
    r1, r2, r3 = sp.symbols('r1 r2 r3')
    
    # 考慮一個非正交的度規 (例如坐標變換)
    # 為了簡化測試，我們仍使用圓柱坐標的度規，但強調這是 g_ij 矩陣
    cyl_coords = [r1, r2, r3]
    cyl_g_matrix = sp.Matrix([
        [1, 0, 0], 
        [0, r1**2, 0], 
        [0, 0, 1]
    ])
    cyl_metric = Metric(cyl_g_matrix, cyl_coords)
    
    print("## 範例 2: 圓柱坐標系 (r1, r2, r3) - 傳入通用度規")
    print(f"坐標變數: {cyl_metric.coords}")
    print(f"度規矩陣 g: {cyl_metric.g}")
    print(f"sqrt(|det(g)|): {cyl_metric.sqrt_det_g}")
    
    # 散度測試：逆變向量場 F^i = [1/r1, 0, 0]
    F_cyl_contravariant = sp.Matrix([1/r1, 0, 0])
    div_F_cyl = divergence(F_cyl_contravariant, metric=cyl_metric)
    print(f"逆變向量場 F^i: {F_cyl_contravariant}")
    print(f"散度 ∇·F: {div_F_cyl}")
    # 預期結果: (1/r1) * d/dr1(r1 * (1/r1)) = (1/r1) * d/dr1(1) = 0
    
    # 旋度測試：協變向量場 A_i = [r1*r2, 0, 0]
    A_cyl_covariant = sp.Matrix([r1*r2, 0, 0])
    curl_A_cyl = curl(A_cyl_covariant, metric=cyl_metric)
    print(f"協變向量場 A_i: {A_cyl_covariant}")
    print(f"旋度 ∇×A (協變分量): {curl_A_cyl}")
    # 預期 (∇×A)^3 = (1/r1) * (∂A2/∂r1 - ∂A1/∂r2) = (1/r1) * (0 - r1) = -1
    # Curl_covariant_3 = g_33 * (Curl)^3 = 1 * (-1) = -1
    
    print("\n--------------------------------------------------")