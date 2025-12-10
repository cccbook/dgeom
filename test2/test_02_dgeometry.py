import sympy as sp
import numpy as np
import math
import pytest
from dgeom.sym import *

# --------------------------------------------------
# I. 基礎符號設置
# --------------------------------------------------

x, y, z = sp.symbols('x y z')
coords_euc = [x, y, z]

r, theta, phi = sp.symbols('r theta phi')
coords_sph = [r, theta, phi]


# --------------------------------------------------
# II. 測試函式
# --------------------------------------------------
# 引入 SymPy 符號 t 用於參數化

t = sp.symbols('t')

# ----------- 微分幾何版的向量微積分 dvcalculus.py 的測試 ---------------
def test_dvcalculus_curl_of_gradient_is_zero():
    """
    ### 🧪 驗證 vcalculus.py：梯度的旋度為零
    數學公式: $\nabla \times (\nabla f) = \mathbf{0}$
    """
    f = x**2 * y * sp.cos(z)
    
    # [修正 1] 將 Metric 包裝成 HodgeMetric 以支援 sharp/flat/star 運算
    h_metric = HodgeMetric(EUCLIDEAN_METRIC.g, EUCLIDEAN_METRIC.coords)
    
    grad_f = d_gradient(f, h_metric)      
    curl_grad_f = d_curl(grad_f, h_metric) 

    assert sp.simplify(curl_grad_f) == sp.zeros(3, 1), \
        r"∇ × (∇f) 應為零向量"


def test_dvcalculus_curl_of_gradient_spherical():
    """
    ### 🧪 驗證 dvcalculus.py：球坐標下的 $\nabla \times (\nabla f) = \mathbf{0}$
    """
    metric = SPHERICAL_METRIC
    f = r**2 * sp.cos(theta) * sp.sin(phi)

    # 這裡假設 d_gradient 回傳的是 Form 物件，其 .op 是係數
    grad_f_cov = d_gradient(f, metric) 
    curl_grad_f_cov = d_curl(grad_f_cov, metric) 

    # [新修正 4] d_curl 回傳的是 SymPy Matrix，無須存取 .components 屬性
    assert sp.simplify(curl_grad_f_cov) == sp.zeros(3, 1), \
        r"在球坐標下，d_curl(d_gradient(f)) 應為零向量"


def test_dvector_exterior_derivative_dd_is_zero():
    """
    ### 🧪 驗證 dvector.py：外微分的平方為零
    數學公式: $d(d(\omega)) = 0$
    """
    f = x*y*z
    omega_0 = Form(0, f) 
    
    d_omega_0 = d_operator(omega_0)  
    dd_omega_0 = d_operator(d_omega_0) 
    
    expected_coeffs = 0 # 更改為零純量，以匹配程式庫優化回傳的結果 (不是零矩陣)
    
    # [新修正 3] .op 是一個函式，必須呼叫它 dd_omega_0.op() 才能取得係數矩陣
    assert sp.simplify(dd_omega_0.op()) == expected_coeffs, \
        r"外微分的平方 $d(d(\omega))$ 的所有分量應為零"


def test_hodge_flat_sharp_inversion():
    """
    ### 🧪 驗證 hodge.py：指標升降的逆運算
    """
    # 這裡正確使用了 HodgeMetric
    metric = HodgeMetric(EUCLIDEAN_METRIC.g, EUCLIDEAN_METRIC.coords)

    # V 是一個 TangentVector 物件
    V = TangentVector(sp.Matrix([x**2, y, sp.cos(z)]), coords_euc, name='V') 
    
    V_flat = metric.flat(V)     
    V_sharp = metric.sharp(V_flat)
    
    V_orig_comps = V.components
    V_sharp_comps = V_sharp.components
    
    assert sp.simplify(V_sharp_comps - V_orig_comps) == sp.zeros(3, 1), \
        r"指標升降運算應為逆運算"

