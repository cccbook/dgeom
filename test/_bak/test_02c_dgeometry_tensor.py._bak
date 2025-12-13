import pytest
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

from dgeom.sym import *

# ===================================================================
# 測試案例
# ===================================================================

def test_dvcalculus_curl_of_gradient_is_zero():
    """
    ### 🧪 驗證：梯度的旋度為零 (Euclidean)
    """
    # 1. 使用新版工廠函數取得 TensorMetric
    tm = euclidean_metric()
    x, y, z = tm.coords 

    # 2. 定義純量場 f
    f = x**2 * y * sp.cos(z)
    
    # 3. 計算 (新版 d_gradient/d_curl 支援 TensorMetric)
    # 結果是 GeometricTensor (Rank 1)
    grad_f = d_gradient(f, tm)      
    curl_grad_f = d_curl(grad_f, tm) 

    # 4. 驗證
    # [修正] GeometricTensor 不能直接與 sp.zeros(3,1) 比較
    # 需檢查其 .data (NDimArray) 的所有元素
    for val in np.array(curl_grad_f.data).flatten():
        assert sp.simplify(val) == 0, f"分量 {val} 應為 0"


def test_dvcalculus_curl_of_gradient_spherical():
    """
    ### 🧪 驗證：球坐標下的梯度旋度為零
    """
    # 1. 取得新版球坐標度規
    tm = spherical_metric()
    r, theta, phi = tm.coords

    # 2. 定義純量場 f
    f = r**2 * sp.cos(theta) * sp.sin(phi)

    # 3. 計算
    # 注意: 新版 d_gradient 回傳協變向量 ([-1])
    # 新版 d_curl 接受協變向量並回傳協變向量
    grad_f = d_gradient(f, tm) 
    curl_grad_f = d_curl(grad_f, tm) 

    # 4. 驗證
    for val in np.array(curl_grad_f.data).flatten():
        assert sp.simplify(val) == 0, f"球坐標下 Curl(Grad) 分量 {val} 應為 0"


def test_dvector_exterior_derivative_dd_is_zero():
    """
    ### 🧪 驗證：外微分的平方為零 d(d(omega)) = 0
    """
    tm = euclidean_metric()
    x, y, z = tm.coords
    
    # 0-form
    f = x*y*z
    omega_0 = Form(0, f) 
    
    # d(d(f)) -> 2-form
    d_omega_0 = d_operator(omega_0)  
    dd_omega_0 = d_operator(d_omega_0) 
    
    # [修正] 新版 Form 是一個運算子，dd_omega_0 是 2-form。
    # 要驗證它為零，我們必須將其作用在任意兩個向量上，看結果是否為 0。
    # 或者檢查其內部邏輯 (但那是實作細節)。標準做法是代入向量。
    
    v1 = TangentVector([1, 0, 0], tm.coords)
    v2 = TangentVector([0, 1, z], tm.coords)
    
    result = dd_omega_0(v1, v2)
    
    assert sp.simplify(result) == 0


def test_hodge_flat_sharp_inversion():
    """
    ### 🧪 驗證：指標升降 (Musical Isomorphisms)
    使用新版 HodgeMetric (繼承自 TensorMetric)
    """
    # 1. 準備度規
    tm = euclidean_metric()
    x, y, z = tm.coords
    
    # 2. 建立 HodgeMetric
    # 新版 HodgeMetric 接受 data (NDimArray)
    h_metric = HodgeMetric(tm.data, tm.coords)

    # 3. 建立向量 V (新版 TangentVector)
    # 輸入可以是 list，TangentVector 會自動轉 NDimArray
    V = TangentVector([x**2, y, sp.cos(z)], tm.coords, name='V') 
    
    # 4. 執行升降運算
    V_flat = h_metric.flat(V)     # Vector -> 1-Form
    V_sharp = h_metric.sharp(V_flat) # 1-Form -> Vector
    
    # 5. 驗證逆運算
    # [修正] 新版 TangentVector 使用 .data (NDimArray)，而非 .components (Matrix)
    diff_data = V_sharp.data - V.data
    
    for val in np.array(diff_data).flatten():
        assert sp.simplify(val) == 0


# ===================================================================
# 核心測試：TensorMetric 內建的測地線功能
# ===================================================================

def test_geodesic_equations_generation():
    """
    驗證 TensorMetric 能正確生成測地線微分方程 (符號)。
    """
    theta, phi = sp.symbols('theta phi', real=True)
    coords = [theta, phi]
    g_data = sp.diag(1, sp.sin(theta)**2)
    
    tm = TensorMetric(g_data, coords)
    
    tau = sp.Symbol('tau')
    eqs = tm.get_geodesic_equations(param_var=tau)
    
    # TensorMetric 回傳形式: Eq(acc, -gamma_term)
    # 即 theta'' = RHS
    theta_func = sp.Function('theta')(tau)
    phi_func = sp.Function('phi')(tau)
    
    theta_rhs = eqs[0].rhs
    
    # 理論值: theta'' = sin(theta)cos(theta) * (phi')^2
    expected_rhs = sp.sin(theta_func) * sp.cos(theta_func) * sp.diff(phi_func, tau)**2
    
    assert sp.simplify(theta_rhs - expected_rhs) == 0

@pytest.mark.skipif(not pytest.importorskip("scipy"), reason="需要 scipy")
def test_geodesic_bvp_numerical_solution():
    """
    數值驗證：球面上的測地線 (大圓)。
    """
    theta, phi = sp.symbols('theta phi', real=True)
    coords = [theta, phi]
    g_data = sp.diag(1, sp.sin(theta)**2)
    tm = TensorMetric(g_data, coords)
    
    # 沿經線走 (phi 固定)
    start = [0.1, 0.0]
    end = [np.pi/2, 0.0]
    
    path = tm.solve_geodesic_bvp(start, end, num_points=20)
    
    thetas = path[0]
    phis = path[1]
    
    # 驗證 phi 保持 0
    assert np.allclose(phis, 0.0, atol=1e-4)
    
    # 驗證 theta 線性增加
    theta_diffs = np.diff(thetas)
    assert np.std(theta_diffs) < 1e-4

# ===================================================================
# 視覺化測試
# ===================================================================

if __name__ == "__main__":
    print("正在執行球面測地線視覺化...")
    theta, phi = sp.symbols('theta phi', real=True)
    g_data = sp.diag(1, sp.sin(theta)**2)
    tm = TensorMetric(g_data, [theta, phi])
    
    start = [0.2, 0.0]
    end = [np.pi/2, np.pi/2] # 斜向走
    
    try:
        path = tm.solve_geodesic_bvp(start, end, num_points=50)
        thetas = path[0]
        phis = path[1]

        X = np.sin(thetas) * np.cos(phis)
        Y = np.sin(thetas) * np.sin(phis)
        Z = np.cos(thetas)

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
        x_sphere = np.cos(u)*np.sin(v)
        y_sphere = np.sin(u)*np.sin(v)
        z_sphere = np.cos(v)
        ax.plot_wireframe(x_sphere, y_sphere, z_sphere, color="gray", alpha=0.1)
        
        ax.plot(X, Y, Z, color='r', linewidth=3, label='Geodesic')
        ax.scatter([X[0]], [Y[0]], [Z[0]], color='g', s=100)
        ax.scatter([X[-1]], [Y[-1]], [Z[-1]], color='b', s=100)
        
        ax.legend()
        plt.show()
        
    except Exception as e:
        print(f"執行錯誤: {e}")