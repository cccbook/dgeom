import pytest
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from dgeom.sym import *

def test_general_stoke():
    r"""
    ### 🧪 驗證廣義史托克定理 (Generalized Stokes' Theorem)
    數學公式: $\int_{\Omega} d\omega = \int_{\partial\Omega} \omega$
    
    設定:
      - 空間: 3D 歐氏空間 (x, y, z)
      - 流形 $\Omega$: 參數化曲面 patch, map(u, v) -> (u, v, u*v)
        (這是一個雙曲拋物面的一部分，定義域為單位正方形 [0,1]x[0,1])
      - 微分形式 $\omega$: 1-Form, $\omega = z dx$
        (其外微分 $d\omega = dz \wedge dx$)
    """
    # 1. 定義坐標與參數
    x, y, z = sp.symbols('x y z', real=True)
    coords = [x, y, z]
    u, v = sp.symbols('u v', real=True)
    
    # 2. 定義微分形式 omega = z dx
    # Form 的 evaluator 接受一個切向量 V，回傳數值
    # omega(V) = z * (V 的 x 分量)
    def omega_func(V):
        # V.coords 對應 [x, y, z]
        # V.data   對應 [Vx, Vy, Vz]
        z_val = V.coords[2] # 符號 z
        Vx = V.data[0]      # dx(V)
        return z_val * Vx

    omega = Form(1, omega_func) # 1-Form
    
    # 3. 計算外微分 d(omega) -> 2-Form
    d_omega = d_operator(omega)
    
    # 4. 定義流形 (2D Parametric Patch in 3D)
    # 映射: x=u, y=v, z=u*v
    map_func = lambda params: [params[0], params[1], params[0] * params[1]]
    
    # 參數範圍: u in [0, 1], v in [0, 1]
    patch = ParametricPatch([u, v], [(0, 1), (0, 1)], map_func)
    
    # 5. 計算 LHS: 區域積分 \int_Omega d(omega)
    # integrate_form 會自動處理 pullback (代入 z=uv, dz=vdu+udv ...)
    lhs_volume_integral = integrate_form(d_omega, patch, coords)
    
    # 6. 計算 RHS: 邊界積分 \int_{partial Omega} omega
    # ParametricPatch.get_boundaries() 會回傳 4 個 1D 邊界及其定向符號
    # 邊界分別對應 u=0, u=1, v=0, v=1 的四條曲線
    rhs_boundary_integral = 0
    boundaries = patch.get_boundaries()
    
    for boundary_domain, sign in boundaries:
        # 對每個邊界進行線積分
        val = integrate_form(omega, boundary_domain, coords)
        rhs_boundary_integral += sign * val
        
    # 7. 驗證
    print(f"\n[General Stokes] Volume Integral (d_omega): {lhs_volume_integral}")
    print(f"[General Stokes] Boundary Integral (omega): {rhs_boundary_integral}")
    
    assert sp.simplify(lhs_volume_integral - rhs_boundary_integral) == 0, \
        f"廣義史托克定理驗證失敗: LHS={lhs_volume_integral}, RHS={rhs_boundary_integral}"
        
# ===================================================================
# 外微分版的向量微積分 (Vector Calculus based on d-operator)
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
# 測地線 (Geodesic) - 符號與數值
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