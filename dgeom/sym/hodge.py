import sympy as sp
import itertools
from functools import reduce
from dvector import TangentVector, Form, d, lie_bracket
# ==========================================
# Part 2: 黎曼幾何與通用 Hodge Star (Full Impl)
# ==========================================

class Metric:
    def __init__(self, g_matrix, coords):
        self.g = sp.Matrix(g_matrix)
        self.g_inv = self.g.inv()
        self.coords = coords
        self.dim = len(coords)
        self.det_g = sp.det(self.g)
        self.sqrt_det_g = sp.sqrt(sp.Abs(self.det_g))

    def get_basis_vector(self, i):
        comp = [0] * self.dim
        comp[i] = 1
        return TangentVector(comp, self.coords, name=f"e_{i}")

    def flat(self, vector):
        """Vector -> 1-Form (Lower indices)"""
        coeffs = self.g * vector.components
        def evaluator(v_target):
            return coeffs.dot(v_target.components)
        return Form(1, evaluator)

    def sharp(self, one_form):
        """1-Form -> Vector (Raise indices)"""
        if one_form.k != 1:
            raise ValueError("Sharp only for 1-forms")
        omega_components = []
        for i in range(self.dim):
            basis_vec = self.get_basis_vector(i)
            omega_components.append(one_form(basis_vec))
        omega_vec = sp.Matrix(omega_components)
        vec_components = self.g_inv * omega_vec
        return TangentVector(vec_components, self.coords, name="Sharp(w)")

    def star(self, form):
        """
        通用 Hodge Star Operator (*): k-Form -> (n-k)-Form
        完全實作版本，無 pass
        """
        k = form.k
        n = self.dim
        target_k = n - k
        
        # 1. 提取輸入 Form 的所有分量 omega_K
        # 我們只提取 K 為嚴格遞增的基底組合 (Basis Combinations)
        source_bases = list(itertools.combinations(range(n), k))
        omega_comps = {} 
        for K in source_bases:
            vectors = [self.get_basis_vector(i) for i in K]
            omega_comps[K] = form(*vectors)

        # 2. 預計算輸出 Form 對應於每個基底 dx^J 的係數 C_J
        # 目標: eta = sum (C_J * dx^J)
        target_bases = list(itertools.combinations(range(n), target_k))
        target_coeffs = {}

        # 為了計算 C_J，我們使用線性疊加原理：
        # Star(omega) = sum_{K} omega_K * Star(dx^K)
        # 我們需要計算 Star(dx^K) 在基底 dx^J 上的分量。
        
        # 遍歷每一個目標基底 J (例如 (dy, dz))
        for J in target_bases:
            total_coeff_J = 0
            
            # 遍歷每一個來源基底 K (例如 (dx))
            for K in source_bases:
                w_val = omega_comps[K]
                if w_val == 0: continue
                
                # 計算 Star(dx^K) 在方向 J 上的投影係數
                # 公式核心: eps_{I...J} * g^{I K}
                # I 是 summation indices (長度 k)，K 是來源固定指標
                
                # 因為 g^{ij} 會混合指標，我們必須遍歷所有可能的 I (長度 k，順序重要)
                # 使用 product 產生所有可能的 index 排列 (i1, i2, ..., ik)
                all_I = itertools.product(range(n), repeat=k)
                
                term_sum = 0
                for I_tuple in all_I:
                    # 1. 檢查 Levi-Civita symbol: epsilon_{i1...ik j1...jk}
                    # 合併指標 tuple: I + J
                    full_indices = I_tuple + J
                    lev_val = sp.LeviCivita(*full_indices)
                    
                    if lev_val == 0: continue
                    
                    # 2. 計算度規收縮: g^{i1 k1} * g^{i2 k2} * ...
                    # K 是當前來源基底 (k1, k2...)
                    metric_contraction = 1
                    for idx in range(k):
                        # g_inv[row, col]
                        metric_contraction *= self.g_inv[I_tuple[idx], K[idx]]
                    
                    term_sum += lev_val * metric_contraction
                
                # 累加貢獻: omega_K * (該基底轉換後的係數)
                total_coeff_J += w_val * term_sum

            # 乘上體積形式係數 sqrt(|g|)
            # 注意：標準公式通常有 1/k!，但因為我們 K 是遍歷「排序後的基底」，
            # 而 I 是遍歷「全排列」，這中間的計數已經自然抵銷。
            # 唯一需要注意的是如果定義不同。但在標準基底展開下，這樣是正確的。
            target_coeffs[J] = sp.simplify(total_coeff_J * self.sqrt_det_g)

        # 3. 定義輸出 Form 的評估函數
        def star_evaluator(*vectors):
            if len(vectors) != target_k:
                raise ValueError(f"Output form expects {target_k} vectors")
            
            # 將輸入向量組合成矩陣 (col vectors)
            V_matrix = sp.Matrix([v.components.T for v in vectors]).T
            
            result = 0
            # 最終值 = Sum_J ( C_J * dx^J(V1...Vm) )
            # dx^J(V) 就是 V 矩陣在列 J 上的子行列式
            for J, coeff in target_coeffs.items():
                if coeff == 0: continue
                
                # 取出對應的列 (SymPy Matrix slicing)
                # J 是一個 tuple list，例如 (1, 2) 代表取第1, 2列
                sub_matrix = V_matrix[list(J), :] 
                det_val = sub_matrix.det()
                
                result += coeff * det_val
                
            return sp.simplify(result)

        return Form(target_k, star_evaluator)

# ==========================================
# Part 3: 向量微積分 (Gradient, Curl, Divergence)
# ==========================================

def gradient(f, metric): # Grad f = (df)^#
    return metric.sharp(d(Form(0, f)))

def curl(vector, metric): # Curl V = (* d V^b)^#
    if metric.dim != 3: raise ValueError("Curl only for 3D")
    v_flat = metric.flat(vector)
    dv = d(v_flat)
    star_dv = metric.star(dv)
    return metric.sharp(star_dv)

def divergence(vector, metric): # Div V = * d (* V^b)
    v_flat = metric.flat(vector)
    star_v = metric.star(v_flat)
    d_star_v = d(star_v)
    res = metric.star(d_star_v)
    return res() # 0-form to scalar

# ==========================================
# Part 4: 測試與驗證
# ==========================================

def test_hodge():
    print("=== Full Hodge Star Implementation Test ===")
    
    x, y, z = sp.symbols('x y z')
    coords = [x, y, z]
    
    # 1. 歐幾里得度規 (Euclidean)
    print("\n[Case 1] Euclidean Metric")
    g_euclid = sp.eye(3)
    metric_e = Metric(g_euclid, coords)
    
    # 測試旋度
    V = TangentVector([-y, x, 0], coords, "V") # 剛體旋轉
    c = curl(V, metric_e)
    print(f"Curl([-y, x, 0]) = {c.components.T} (Expected: [0, 0, 2])")
    
    # 2. 對角非單位度規 (例如縮放過的空間)
    print("\n[Case 2] Scaled Metric (ds^2 = 4dx^2 + dy^2 + dz^2)")
    # g_xx = 4, 意味著 x 方向長度是被放大的，對應的梯度應該變小
    g_scaled = sp.diag(4, 1, 1) 
    metric_s = Metric(g_scaled, coords)
    
    f = x
    grad_f = gradient(f, metric_s)
    # df = dx. Grad = g^{ij} d_j f. 
    # g^{xx} = 1/4. So Grad(x) should be [1/4, 0, 0]
    print(f"Gradient(x) in scaled metric: {grad_f.components.T} (Expected: [1/4, 0, 0])")

    # 3. 測試散度 (Divergence)
    W = TangentVector([x, y, z], coords, "W")
    div_w = divergence(W, metric_e)
    print(f"Div([x, y, z]) = {div_w} (Expected: 3)")

    # 4. 驗證 Hodge Star 本身: *dx
    # 在歐氏空間 *dx = dy^dz
    dx = metric_e.flat(TangentVector([1, 0, 0], coords))
    star_dx = metric_e.star(dx)
    
    # 測試作用在 dy, dz 上
    dy = TangentVector([0, 1, 0], coords)
    dz = TangentVector([0, 0, 1], coords)
    val = star_dx(dy, dz)
    print(f"*(dx) applied to (dy, dz): {val} (Expected: 1)")
    
    # 測試作用在 dx, dy 上 (應該是 0)
    dx_vec = TangentVector([1, 0, 0], coords)
    val_zero = star_dx(dx_vec, dy)
    print(f"*(dx) applied to (dx, dy): {val_zero} (Expected: 0)")

def test_ddf_hodge_zero():
    print("\n=== 向量微積分恆等式驗證 (d^2 = 0) ===")
    
    # 1. 設置環境 (3D 歐幾里得空間)
    x, y, z = sp.symbols('x y z')
    coords = [x, y, z]
    metric = Metric(sp.eye(3), coords)
    
    # ==========================================
    # 測試 1: 梯度的旋度 (Curl of Gradient)
    # Identity: Curl(Grad f) = 0
    # ==========================================
    print("\n[Test 1] Curl(Gradient(f)) == 0 ?")
    
    # 定義一個非平凡的純量函數 (包含混合項)
    f = x**3 * y + sp.sin(z) * x + sp.exp(y)
    print(f"  f(x,y,z) = {f}")
    
    # 計算梯度
    grad_f = gradient(f, metric)
    print(f"  Grad(f)  = {grad_f.components.T}")
    
    # 計算旋度
    curl_grad_f = curl(grad_f, metric)
    print(f"  Curl(Grad(f)) = {curl_grad_f.components.T}")
    
    # 驗證每個分量是否為 0
    is_zero_vector = all(sp.simplify(c) == 0 for c in curl_grad_f.components)
    
    if is_zero_vector:
        print("  => 驗證成功 (PASSED)")
    else:
        print("  => 驗證失敗 (FAILED)")

    # ==========================================
    # 測試 2: 旋度的散度 (Divergence of Curl)
    # Identity: Div(Curl V) = 0
    # ==========================================
    print("\n[Test 2] Divergence(Curl(V)) == 0 ?")
    
    # 定義一個非平凡的向量場
    # V = [yz, x^2, xy]
    V = TangentVector([y*z, x**2, x*y], coords, name="V")
    print(f"  V        = {V.components.T}")
    
    # 計算旋度
    curl_V = curl(V, metric)
    print(f"  Curl(V)  = {curl_V.components.T}")
    
    # 計算散度
    div_curl_V = divergence(curl_V, metric)
    print(f"  Div(Curl(V)) = {div_curl_V}")
    
    # 驗證結果是否為 0
    if sp.simplify(div_curl_V) == 0:
        print("  => 驗證成功 (PASSED)")
    else:
        print("  => 驗證失敗 (FAILED)")

    # ==========================================
    # 額外測試: 一般度規下的驗證 (General Metric)
    # 恆等式在任何黎曼流形上都應成立
    # ==========================================
    print("\n[Test 3] 非歐幾里得度規下的驗證")
    # 定義一個對角度規 g = diag(x^2, 1, 1) (類似某種變形空間)
    g_matrix = sp.diag(x**2, 1, 1)
    metric_gen = Metric(g_matrix, coords)
    print(f"  Metric g = diag(x^2, 1, 1)")
    
    # 測試 Div(Curl V) 在此度規下
    # 注意：這裡的 Curl 和 Div 定義包含了 sqrt(|g|) 和 g_inv，計算非常繁瑣
    V_gen = TangentVector([z, 0, x], coords) # 簡單一點的向量，避免計算跑太久
    
    res = divergence(curl(V_gen, metric_gen), metric_gen)
    print(f"  Div(Curl([z, 0, x])) = {sp.simplify(res)}")
    
    if sp.simplify(res) == 0:
        print("  => 驗證成功 (PASSED)")
    else:
        print("  => 驗證失敗 (FAILED)")

if __name__ == "__main__":
    test_hodge()
    test_ddf_hodge_zero()