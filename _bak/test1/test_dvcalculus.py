from dgeom.sym import d_gradient, d_divergence, d_curl, Metric, EUCLIDEAN_METRIC
import sympy as sp
# --------------------------------------------------
# IV. 測試範例 (Test Cases)
# --------------------------------------------------

if __name__ == "__main__":
    print("🚀 向量微積分 (vcalculus) 模組測試：通用張量版本")
    print("--------------------------------------------------")

    # --- 1. 歐幾里得度規 (直角坐標 x, y, z) 測試 ---
    print("## 範例 1: 歐幾里得度規 (直角坐標) - 預設參數")
    x, y, z = EUCLIDEAN_METRIC.coords # euclidean_coords
    # 純量場 f = x*y*z
    f_euc = x * y * z
    grad_f_euc = d_gradient(f_euc) 
    print(f"純量場 f = {f_euc}")
    print(f"梯度 ∇f (協變分量): {grad_f_euc}")

    # 向量場 F (直角坐標系下，協變/逆變分量相同): F^i = F_i = [x, y, 0]
    F_euc = sp.Matrix([x, y, 0]) 
    
    div_F_euc = d_divergence(F_euc) 
    curl_F_euc = d_curl(F_euc)       
    
    print(f"向量場 F (分量) = {F_euc}")
    print(f"散度 ∇·F: {div_F_euc}")
    print(f"旋度 ∇×F (協變分量): {curl_F_euc}")

    # --------------------------------------------------
    print("\n" + "---" * 15 + "\n")
    
    # --- 2. 圓柱坐標系 (非正交，非預設度規) 測試 ---
    # 使用 r1, r2, r3 表示坐標 (rho, phi, z)
    r1, r2, r3 = sp.symbols('r1 r2 r3')
    
    # 圓柱坐標的度規 g_ij = diag(1, r1^2, 1)
    cyl_coords = [r1, r2, r3]
    cyl_g_matrix = sp.Matrix([
        [1, 0, 0], 
        [0, r1**2, 0], 
        [0, 0, 1]
    ])
    cyl_metric = Metric(cyl_g_matrix, cyl_coords)
    
    print("## 範例 2: 圓柱坐標系 (r1, r2, r3) - 通用度規測試")
    print(f"坐標變數: {cyl_metric.coords}")
    print(f"度規矩陣 g: {cyl_metric.g}")
    print(f"sqrt(|det(g)|): {cyl_metric.sqrt_det_g}")
    
    # 散度測試：逆變向量場 F^i = [1/r1, 0, 0]
    F_cyl_contravariant = sp.Matrix([1/r1, 0, 0])
    div_F_cyl = d_divergence(F_cyl_contravariant, metric=cyl_metric)
    print(f"逆變向量場 F^i: {F_cyl_contravariant}")
    print(f"散度 ∇·F: {div_F_cyl}")
    
    # 旋度測試：協變向量場 A_i = [r1*r2, 0, 0]
    A_cyl_covariant = sp.Matrix([r1*r2, 0, 0])
    curl_A_cyl = d_curl(A_cyl_covariant, metric=cyl_metric)
    print(f"協變向量場 A_i: {A_cyl_covariant}")
    print(f"旋度 ∇×A (協變分量): {curl_A_cyl}")
    
    print("\n--------------------------------------------------")

    # --------------------------------------------------
    print("\n" + "---" * 15 + "\n")
    
    # --- V. 向量恆等式測試 (通用張量) ---
    print("## 範例 3: 向量微積分恆等式驗證 (圓柱坐標系)")
    
    # 定義圓柱坐標系 (r1, r2, r3) -> (rho, phi, z)
    r1, r2, r3 = sp.symbols('r1 r2 r3')
    cyl_coords = [r1, r2, r3]
    cyl_g_matrix = sp.Matrix([
        [1, 0, 0], 
        [0, r1**2, 0], 
        [0, 0, 1]
    ])
    cyl_metric = Metric(cyl_g_matrix, cyl_coords)

    ### 恆等式 1: 梯度的旋度為零 (Curl of the Gradient) ###
    
    # 選擇一個純量場 f
    f_test = r1**2 * sp.cos(r2) * r3 
    
    # 步驟 1: 計算梯度 ∇f (結果是協變分量)
    grad_f = d_gradient(f_test, metric=cyl_metric)
    
    # 步驟 2: 計算梯度的旋度 ∇ × (∇f)
    curl_grad_f = d_curl(grad_f, metric=cyl_metric)
    
    print("\n--- 1. ∇ × (∇f) = 0 驗證 ---")
    print(f"測試純量場 f: {f_test}")
    print(f"梯度 ∇f (協變): {grad_f}")
    print(f"梯度的旋度 ∇×(∇f): {curl_grad_f}")
    
    # 確認結果是否為零向量
    is_curl_grad_zero = curl_grad_f.is_zero
    print(f"結果是否為零向量: {is_curl_grad_zero}")
    
    ### 恆等式 2: 旋度的散度為零 (Divergence of the Curl) ###
    
    # 選擇一個協變向量場 A (例如，來自某個物理潛勢)
    # A_i = [r1*r2, 0, r3^2]
    A_covariant_test = sp.Matrix([r1 * r2, 0, r3**2])
    
    # 步驟 1: 計算旋度 ∇ × A (結果是協變分量)
    curl_A_covariant = d_curl(A_covariant_test, metric=cyl_metric)
    
    # ⚠️ 轉換：散度函數 `d_d_divergence` 需要**逆變分量**，
    #   故我們必須將協變的 (∇×A)_i 轉換為逆變的 (∇×A)^i
    curl_A_contravariant = cyl_metric.g_inv * curl_A_covariant
    
    # 步驟 2: 計算旋度的散度 ∇ ⋅ (∇ × A)
    div_curl_A = d_divergence(curl_A_contravariant, metric=cyl_metric)
    
    print("\n--- 2. ∇ ⋅ (∇ × A) = 0 驗證 ---")
    print(f"測試協變向量場 A_i: {A_covariant_test}")
    print(f"旋度 ∇×A (協變): {curl_A_covariant}")
    print(f"旋度的散度 ∇·(∇×A): {sp.simplify(div_curl_A)}")
    
    # 確認結果是否為零
    is_div_curl_zero = sp.simplify(div_curl_A) == 0
    print(f"結果是否為零: {is_div_curl_zero}")
    
    print("\n--------------------------------------------------")
