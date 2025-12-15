import pytest
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from dgeom.sym import *

# ===================================================================
# 測試: 外微分 (Exterior Derivative)
# ===================================================================

def test_ddf_is_zero():
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

