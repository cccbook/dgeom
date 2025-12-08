# vcalculus.py (版本 3.0 - 簡化函式名與歐幾里得預設)
# -------------------------------------------------------------
# 描述：在正交曲線坐標系下，實作梯度、散度與旋度運算。
# 預設使用歐幾里得度規 (直角坐標系)。
# -------------------------------------------------------------

from sympy import symbols, diff, Matrix, simplify, sqrt

# --------------------------------------------------
# 基礎定義：直角坐標系 (歐幾里得度規)
# --------------------------------------------------

# 宣告直角坐標變數
x, y, z = symbols('x y z')
euclidean_coords = [x, y, z]

# 歐幾里得度規矩陣 g_ij = diag(1, 1, 1)
euclidean_g_matrix = Matrix([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])

# --------------------------------------------------
# Metric 類別 (度規數據封裝)
# --------------------------------------------------

class Metric:
    """
    用於在三維正交曲線坐標系下，封裝坐標變數和度規信息的類別。
    h_factors: 標度因子 [h1, h2, h3]，其中 h_i = sqrt(g_ii)。
    H: 體積元素因子 H = h1 * h2 * h3。
    """
    
    def __init__(self, g_matrix, coords):
        if g_matrix.shape != (3, 3) or len(coords) != 3:
            raise ValueError("度規矩陣必須是 3x3，且坐標變數必須有 3 個。")
            
        self.g_matrix = g_matrix
        self.coords = coords
        
        # 提取標度因子 h_i = sqrt(g_ii)
        h_factors = [sqrt(g_matrix[i, i]) for i in range(3)]
        self.h_factors = h_factors
        
        # 體積元素因子 H
        self.H = simplify(h_factors[0] * h_factors[1] * h_factors[2])

# 建立預設的歐幾里得度規實例
EUCLIDEAN_METRIC = Metric(euclidean_g_matrix, euclidean_coords)

# --------------------------------------------------
# 向量微積分函式 (預設 metric 為歐幾里得)
# --------------------------------------------------

def gradient(f, metric=EUCLIDEAN_METRIC):
    """
    計算純量場 f 的梯度 (Gradient, ∇f)。
    預設 metric 為歐幾里得度規 (直角坐標系)。
    """
    h_factors = metric.h_factors
    coords = metric.coords
    
    grad_components = []
    for i in range(3):
        # 梯度分量: (1/h_i) * (∂f/∂u_i)
        component = (1 / h_factors[i]) * diff(f, coords[i])
        grad_components.append(component)

    return Matrix(grad_components)


def divergence(F, metric=EUCLIDEAN_METRIC):
    """
    計算向量場 F 的散度 (Divergence, ∇ · F)。
    F 必須是物理分量向量 [A1, A2, A3]。
    預設 metric 為歐幾里得度規。
    """
    h1, h2, h3 = metric.h_factors
    u1, u2, u3 = metric.coords
    A1, A2, A3 = F[0], F[1], F[2]
    
    H = metric.H # H = h1*h2*h3
    
    # H_i = H / h_i
    H1 = h2 * h3
    H2 = h1 * h3
    H3 = h1 * h2
    
    # 公式項: ∂/∂u_i (A_i * H_i)
    term1 = diff(A1 * H1, u1)
    term2 = diff(A2 * H2, u2)
    term3 = diff(A3 * H3, u3)
    
    # 散度: (1/H) * (term1 + term2 + term3)
    div_sum = (1 / H) * (term1 + term2 + term3)
        
    return simplify(div_sum)


def curl(F, metric=EUCLIDEAN_METRIC):
    """
    計算向量場 F 的旋度 (Curl, ∇ × F)。
    F 必須是物理分量向量 [A1, A2, A3]。
    預設 metric 為歐幾里得度規。
    """
    u1, u2, u3 = metric.coords
    h1, h2, h3 = metric.h_factors
    A1, A2, A3 = F[0], F[1], F[2]
    
    # 乘以標度因子後的向量分量 (A_i * h_i)
    H_components = [A1 * h1, A2 * h2, A3 * h3]
    
    # 旋度的三個物理分量 (e1, e2, e3 分量)
    
    # 1. e1 分量: (1/(h2*h3)) * [∂/∂u2(A3*h3) - ∂/∂u3(A2*h2)]
    e1_comp = (1 / (h2 * h3)) * (diff(H_components[2], u2) - diff(H_components[1], u3))
    
    # 2. e2 分量: (1/(h1*h3)) * [∂/∂u3(A1*h1) - ∂/∂u1(A3*h3)]
    e2_comp = (1 / (h1 * h3)) * (diff(H_components[0], u3) - diff(H_components[2], u1))
    
    # 3. e3 分量: (1/(h1*h2)) * [∂/∂u1(A2*h2) - ∂/∂u2(A1*h1)]
    e3_comp = (1 / (h1 * h2)) * (diff(H_components[1], u1) - diff(H_components[0], u2))
    
    return Matrix([simplify(e1_comp), simplify(e2_comp), simplify(e3_comp)])


# --------------------------------------------------
# 測試範例
# --------------------------------------------------

if __name__ == "__main__":
    print("🚀 向量微積分 (vcalculus) 模組測試")
    print("--------------------------------------------------")

    # --- 1. 歐幾里得度規 (直角坐標系) 測試 ---
    print("## 範例 1: 歐幾里得度規 (直角坐標 x, y, z) - 預設參數")
    
    # 純量場 f = x^2 * y
    f_euc = x**2 * y
    grad_f_euc = gradient(f_euc) # 不傳入 metric 參數
    print(f"f = {f_euc}")
    print(f"梯度 ∇f: {grad_f_euc}")
    # 預期結果: [2*x*y, x**2, 0]

    # 向量場 F = [x*y, z, 0]
    F_euc = Matrix([x * y, z, 0])
    div_F_euc = divergence(F_euc)
    curl_F_euc = curl(F_euc)
    print(f"F = {F_euc}")
    print(f"散度 ∇·F: {div_F_euc}")
    # 預期結果: d(xy)/dx + d(z)/dy + d(0)/dz = y + 0 + 0 = y
    print(f"旋度 ∇×F: {curl_F_euc}")
    # 預期結果: [d(0)/dy - d(z)/dz, d(xy)/dz - d(0)/dx, d(z)/dx - d(xy)/dy] = [-1, 0, -x]

    # --------------------------------------------------
    print("\n" + "---" * 15 + "\n")
    
    # --- 2. 圓柱坐標系 (非預設度規) 測試 ---
    print("## 範例 2: 圓柱坐標系 (rho, phi, z) - 傳入 metric 參數")
    
    rho, phi, z = symbols('rho phi z')
    cyl_coords = [rho, phi, z]
    cyl_g_matrix = Matrix([[1, 0, 0], [0, rho**2, 0], [0, 0, 1]])
    cyl_metric = Metric(cyl_g_matrix, cyl_coords)
    
    # 向量場 G (物理分量): [0, rho, 0]
    G_cyl = Matrix([0, rho, 0])
    curl_G_cyl = curl(G_cyl, metric=cyl_metric) # 傳入 cyl_metric
    print(f"G = {G_cyl} (圓柱物理分量)")
    print(f"旋度 ∇×G: {curl_G_cyl}")
    # 預期結果: [0, 0, 2]
    
    print("\n--------------------------------------------------")