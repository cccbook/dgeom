import pytest
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from dgeom.sym import *

# ===================================================================
# 測試 1: 向量微積分 (Vector Calculus)
# ===================================================================

def test_calculus_curl_of_gradient_spherical():
    """
    ### 🧪 驗證：球坐標下的梯度旋度為零
    數學公式: ∇ × (∇f) = 0
    
    這個測試非常有價值，因為它同時驗證了：
    1. d_gradient (協變導數)
    2. d_curl (包含 sqrt(g) 和 Levi-Civita 運算)
    3. MetricTensor 在非笛卡爾坐標系的正確性
    """
    # 1. 取得新版球坐標度規
    tm = spherical_metric()
    r, theta, phi = tm.coords

    # 2. 定義純量場 f
    f = r**2 * sp.cos(theta) * sp.sin(phi)

    # 3. 計算
    # grad_f 是協變向量 (1-form)
    grad_f = d_gradient(f, tm) 
    
    # curl_grad_f 是旋度 (通常轉回協變向量以方便比較)
    curl_grad_f = d_curl(grad_f, tm) 

    # 4. 驗證所有分量為 0
    # 注意: MetricTensor 使用 NDimArray，需展開檢查
    for val in np.array(curl_grad_f.data).flatten():
        assert sp.simplify(val) == 0, f"球坐標下 Curl(Grad) 分量應為 0，得到 {val}"


# ===================================================================
# 測試 2: 外微分 (Exterior Derivative)
# ===================================================================

def test_exterior_derivative_squared_is_zero():
    """
    ### 🧪 驗證：外微分的平方為零 d(d(omega)) = 0
    驗證 TangentVector, Form 與 d_operator 的整合。
    """
    tm = euclidean_metric()
    x, y, z = tm.coords
    
    # 0-form (純量場)
    f = x*y*z
    omega_0 = Form(0, f) 
    
    # d(d(f)) -> 2-form
    d_omega_0 = d_operator(omega_0)  
    dd_omega_0 = d_operator(d_omega_0) 
    
    # 驗證算子作用在任意兩個向量場上是否為 0
    v1 = TangentVector([1, 0, 0], tm.coords)
    v2 = TangentVector([0, 1, z], tm.coords)
    
    # 2-form 作用在兩個向量上應回傳純量
    result = dd_omega_0(v1, v2)
    
    assert sp.simplify(result) == 0


# ===================================================================
# 測試 3: 指標升降 (Musical Isomorphisms)
# ===================================================================

def test_hodge_flat_sharp_inversion():
    """
    ### 🧪 驗證：指標升降的可逆性 (Flat vs Sharp)
    驗證 HodgeMetric 是否正確實作了指標升降。
    """
    # 1. 準備度規與向量
    tm = euclidean_metric()
    x, y, z = tm.coords
    
    # 建立 HodgeMetric 介面 (若 MetricTensor 已實作 flat/sharp 可直接用，這裡假設用 HodgeWrapper)
    # 若 dgeom.sym 有直接導出 HodgeMetric，則使用它
    h_metric = HodgeMetric(tm.data, tm.coords)

    # V = x^2 ∂x + y ∂y + cos(z) ∂z
    V = TangentVector([x**2, y, sp.cos(z)], tm.coords, name='V') 
    
    # 2. 執行升降運算
    V_flat = h_metric.flat(V)        # Vector -> 1-Form (降)
    V_restored = h_metric.sharp(V_flat) # 1-Form -> Vector (升)
    
    # 3. 驗證逆運算 (V_restored == V)
    # 檢查數據差異是否為 0
    diff_data = V_restored.data - V.data
    
    for val in np.array(diff_data).flatten():
        assert sp.simplify(val) == 0


# ===================================================================
# 測試 4: 測地線 (Geodesic) - 符號與數值
# ===================================================================

def test_geodesic_equations_symbolic():
    """
    ### 🧪 驗證：測地線方程式的符號生成
    使用 2D 球面 (r=1) 為例。
    """
    theta, phi = sp.symbols('theta phi', real=True)
    coords = [theta, phi]
    g_data = sp.diag(1, sp.sin(theta)**2)
    
    tm = MetricTensor(g_data, coords)
    
    tau = sp.Symbol('tau')
    eqs = tm.get_geodesic_equations(param_var=tau)
    
    # 驗證 theta 分量的方程式
    # 理論值: theta'' - sin(theta)cos(theta)(phi')^2 = 0
    theta_func = sp.Function('theta')(tau)
    phi_func = sp.Function('phi')(tau)
    
    # MetricTensor 回傳 Eq(lhs, rhs) -> lhs - rhs = 0
    # 我們檢查 rhs 是否符合預期 (-Gamma term)
    theta_rhs = eqs[0].rhs
    expected_rhs = sp.sin(theta_func) * sp.cos(theta_func) * sp.diff(phi_func, tau)**2
    
    assert sp.simplify(theta_rhs - expected_rhs) == 0

@pytest.mark.skipif(not pytest.importorskip("scipy"), reason="需要 scipy")
def test_geodesic_bvp_numerical():
    """
    ### 🧪 驗證：數值測地線求解 (BVP)
    驗證球面上的大圓路徑性質。
    """
    theta, phi = sp.symbols('theta phi', real=True)
    coords = [theta, phi]
    g_data = sp.diag(1, sp.sin(theta)**2)
    tm = MetricTensor(g_data, coords)
    
    # 設定邊界：沿著經線走 (phi 固定為 0)
    # 從北極附近 (0.1) 到赤道 (pi/2)
    start = [0.1, 0.0]
    end = [np.pi/2, 0.0]
    
    # 求解
    path = tm.solve_geodesic_bvp(start, end, num_points=21)
    
    thetas = path[0]
    phis = path[1]
    
    # 驗證 1: phi 應該保持恆定 (約為 0)
    assert np.allclose(phis, 0.0, atol=1e-4), "經線測地線的 phi 應保持不變"
    
    # 驗證 2: theta 應該線性增加 (因為度規 g_theta_theta=1 是常數)
    theta_diffs = np.diff(thetas)
    assert np.std(theta_diffs) < 1e-4, "theta 應線性變化 (均勻速度)"


# ===================================================================
# 視覺化 (手動執行用)
# ===================================================================

if __name__ == "__main__":
    print("正在執行球面測地線視覺化...")
    theta, phi = sp.symbols('theta phi', real=True)
    tm = MetricTensor(sp.diag(1, sp.sin(theta)**2), [theta, phi])
    
    # 走一條斜向大圓
    path = tm.solve_geodesic_bvp([0.2, 0.0], [np.pi/2, np.pi/2], num_points=50)
    
    try:
        ts, ps = path[0], path[1]
        X = np.sin(ts) * np.cos(ps)
        Y = np.sin(ts) * np.sin(ps)
        Z = np.cos(ts)

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        
        # 畫網格
        u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
        ax.plot_wireframe(np.cos(u)*np.sin(v), np.sin(u)*np.sin(v), np.cos(v), color="gray", alpha=0.1)
        
        # 畫路徑
        ax.plot(X, Y, Z, 'r-', linewidth=2, label='Geodesic')
        ax.scatter([X[0], X[-1]], [Y[0], Y[-1]], [Z[0], Z[-1]], c=['g', 'b'], s=50)
        ax.legend()
        plt.show()
    except Exception as e:
        print(f"繪圖失敗: {e}")