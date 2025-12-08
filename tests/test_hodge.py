from dgeom.sym.hodge import HodgeMetric, gradient, divergence, curl, TangentVector
import sympy as sp
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
    metric_e = HodgeMetric(g_euclid, coords)
    
    # 測試旋度
    V = TangentVector([-y, x, 0], coords, "V") # 剛體旋轉
    c = curl(V, metric_e)
    print(f"Curl([-y, x, 0]) = {c.components.T} (Expected: [0, 0, 2])")
    
    # 2. 對角非單位度規 (例如縮放過的空間)
    print("\n[Case 2] Scaled HodgeMetric (ds^2 = 4dx^2 + dy^2 + dz^2)")
    # g_xx = 4, 意味著 x 方向長度是被放大的，對應的梯度應該變小
    g_scaled = sp.diag(4, 1, 1) 
    metric_s = HodgeMetric(g_scaled, coords)
    
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
    metric = HodgeMetric(sp.eye(3), coords)
    
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
    metric_gen = HodgeMetric(g_matrix, coords)
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
