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

# ----------- 單純版的向量微積分 dvcalculus.py 的測試 ---------------
t = sp.symbols('t')

def test_vcalculus_curl_of_gradient_is_zero():
    """
    ### 🧪 驗證 v_calculus.py：梯度的旋度為零
    數學公式: $\nabla \times (\nabla f) = \mathbf{0}$
    """
    f = x**2 * y * sp.cos(z)
    
    # 1. 呼叫 v_calculus.py 的 gradient 函式
    grad_f = gradient(f)      # 回傳 TangentVector (假設)
    
    # 2. 呼叫 v_calculus.py 的 curl 函式
    curl_grad_f = curl(grad_f) # 回傳 TangentVector (假設)
    print('curl_grad_f:', curl_grad_f)
    # 3. 取出 components 進行簡化和比較
    # 假設 curl 回傳 TangentVector 物件，該物件有 .components 屬性
    assert sp.simplify(curl_grad_f) == sp.zeros(3, 1), \
        r"∇ × (∇f) 應為零向量 (古典向量微積分)"


def test_vcalculus_divergence_of_curl_is_zero():
    """
    ### 🧪 驗證 v_calculus.py：旋度的散度為零
    數學公式: $\nabla \cdot (\nabla \times \mathbf{F}) = 0$
    """
    # 原始向量場 (SymPy Matrix)
    F_vec = sp.Matrix([x*y**2, y*z**2, z*x**2]) 
    
    curl_F = curl(F_vec)            
    div_curl_F = divergence(curl_F) 

    # divergence 回傳純量 (Scalar, SymPy Expression)，可以直接比較
    assert sp.simplify(div_curl_F) == 0, \
        r"∇ · (∇ × F) 應為零純量"


def test_vcalculus_line_integral_gradient_theorem():
    r"""
    ### 🧪 驗證 v_calculus.py：線積分的梯度定理 (Fundamental Theorem of Calculus)
    數學公式: $\int_{C} \nabla f \cdot d\mathbf{r} = f(\mathbf{r}_B) - f(\mathbf{r}_A)$
    """
    
    # 1. 選擇純量場 f
    f = x**2 * y + sp.sin(z) * 3
    
    # 2. 計算其梯度 $\mathbf{F} = \nabla f$
    F = gradient(f)
    
    # 3. 參數化曲線 C: 從 A=(1, 0, 0) 到 B=(2, 2, $\pi$) 的直線
    # 參數範圍 $t \in [0, 1]$
    t_A, t_B = 0, 1
    
    # 曲線 C 的參數化坐標 $\mathbf{r}(t)$
    # $x(t) = 1 + t(2-1) = 1 + t$
    # $y(t) = 0 + t(2-0) = 2t$
    # $z(t) = 0 + t(\pi-0) = \pi t$
    path_r = sp.Matrix([1 + t, 2 * t, sp.pi * t])
    
    # 4. 理論值: $f(\mathbf{r}_B) - f(\mathbf{r}_A)$
    # B 點坐標: (x=2, y=2, z=$\pi$)
    f_B = f.subs({x: 2, y: 2, z: sp.pi})
    # A 點坐標: (x=1, y=0, z=0)
    f_A = f.subs({x: 1, y: 0, z: 0})
    expected_integral = sp.simplify(f_B - f_A) # $4(2) + 3\sin(\pi) - (1(0) + 3\sin(0)) = 8$
    
    # 5. 實際積分: 呼叫 line_integral 函式
    actual_integral = line_integral(F, path_r, t, t_A, t_B)
    
    # 6. 驗證結果
    assert sp.simplify(actual_integral - expected_integral) == 0, \
        r"梯度定理失敗：線積分 $\int_C \nabla f \cdot d\mathbf{r}$ 不等於 $f(\mathbf{r}_B) - f(\mathbf{r}_A)$"
