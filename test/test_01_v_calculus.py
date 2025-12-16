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

def test_green_theorem():
    r"""
    ### 🧪 驗證格林定理 (Green's Theorem)
    數學公式: $\oint_C (L dx + M dy) = \iint_D (\frac{\partial M}{\partial x} - \frac{\partial L}{\partial y}) dA$
    設定:
      - 區域 D: $xy$ 平面上的單位正方形 $[0,1] \times [0,1]$
      - 向量場: $\mathbf{F} = [-y, x, 0]$ (相當於 $L=-y, M=x$)
    """
    # 1. 定義向量場 F = [-y, x, 0]
    F = sp.Matrix([-y, x, 0])
    
    # 2. 計算 RHS: 雙重積分 (Curl 的 z 分量)
    # curl F = [0, 0, 1 - (-1)] = [0, 0, 2]
    # Integrand = 2
    # Area = 1*1 = 1
    # Expected RHS = 2
    curl_F = curl(F)
    integrand_rhs = curl_F[2] # 取 k 分量
    
    # 使用 SymPy 進行雙重積分 $\int_0^1 \int_0^1 2 dx dy$
    rhs_value = sp.integrate(integrand_rhs, (x, 0, 1), (y, 0, 1))
    
    # 3. 計算 LHS: 沿邊界 C 的線積分 (四段路徑，逆時針)
    # C1: (t, 0), t=0~1
    path_1 = sp.Matrix([t, 0, 0])
    int_1 = line_integral(F, path_1, t, 0, 1)
    
    # C2: (1, t), t=0~1
    path_2 = sp.Matrix([1, t, 0])
    int_2 = line_integral(F, path_2, t, 0, 1)
    
    # C3: (1-t, 1), t=0~1 (向左)
    path_3 = sp.Matrix([1 - t, 1, 0])
    int_3 = line_integral(F, path_3, t, 0, 1)
    
    # C4: (0, 1-t), t=0~1 (向下)
    path_4 = sp.Matrix([0, 1 - t, 0])
    int_4 = line_integral(F, path_4, t, 0, 1)
    
    lhs_value = sp.simplify(int_1 + int_2 + int_3 + int_4)
    
    # 4. 驗證
    assert lhs_value == rhs_value, \
        f"格林定理驗證失敗: LHS(Line)={lhs_value}, RHS(Area)={rhs_value}"

def test_stoke_theorem():
    r"""
    ### 🧪 驗證斯托克斯定理 (Stokes' Theorem)
    數學公式: $\oint_C \mathbf{F} \cdot d\mathbf{r} = \iint_S (\nabla \times \mathbf{F}) \cdot d\mathbf{S}$
    設定:
      - 曲面 S: 位於 $z=0$ 平面的單位正方形 (法向量 $\mathbf{n}=\mathbf{k}$)
      - 向量場: $\mathbf{F} = [2z, 3x, 5y]$ (故意選個三維都有值的)
      - 由於曲面在 xy 平面 (z=0)，F 限制在曲面上變為 [0, 3x, 5y]
    """
    # 1. 定義向量場
    F = sp.Matrix([2*z, 3*x, 5*y])
    
    # 2. 計算 RHS: 曲面積分 $\iint (\nabla \times \mathbf{F}) \cdot \mathbf{k} dA$
    # Curl F
    # x-comp: d(5y)/dy - d(3x)/dz = 5 - 0 = 5
    # y-comp: d(2z)/dz - d(5y)/dx = 2 - 0 = 2
    # z-comp: d(3x)/dx - d(2z)/dy = 3 - 0 = 3
    # Curl F = [5, 2, 3]
    curl_F = curl(F)
    
    # 面法向量 n = [0, 0, 1]
    # Integrand = Curl dot n = 3
    integrand_rhs = curl_F.dot(sp.Matrix([0, 0, 1]))
    
    # 積分區域 x=[0,1], y=[0,1]
    rhs_value = sp.integrate(integrand_rhs, (x, 0, 1), (y, 0, 1))
    
    # 3. 計算 LHS: 線積分 (z=0)
    # 注意: line_integral 會將 path 代入 F，所以雖然 F 有 z，但路徑上 z=0 會被處理
    
    # C1: (t, 0, 0) -> F(t,0,0) dot (1,0,0)
    p1 = sp.Matrix([t, 0, 0])
    i1 = line_integral(F, p1, t, 0, 1) # F=[0, 3t, 0], dr=[1,0,0] -> 0
    
    # C2: (1, t, 0)
    p2 = sp.Matrix([1, t, 0])
    i2 = line_integral(F, p2, t, 0, 1) # F=[0, 3, 5t], dr=[0,1,0] -> 3
    
    # C3: (1-t, 1, 0)
    p3 = sp.Matrix([1-t, 1, 0])
    i3 = line_integral(F, p3, t, 0, 1) # F=[0, 3(1-t), 5], dr=[-1,0,0] -> 0
    
    # C4: (0, 1-t, 0)
    p4 = sp.Matrix([0, 1-t, 0])
    i4 = line_integral(F, p4, t, 0, 1) # F=[0, 0, 5(1-t)], dr=[0,-1,0] -> 0
    
    lhs_value = sp.simplify(i1 + i2 + i3 + i4)
    
    # 4. 驗證 (RHS=3, LHS應為 0+3+0+0 = 3)
    assert lhs_value == rhs_value, \
        f"斯托克斯定理驗證失敗: LHS={lhs_value}, RHS={rhs_value}"


def test_div_theorem():
    r"""
    ### 🧪 驗證散度定理 (Divergence Theorem)
    數學公式: $\oiint_S \mathbf{F} \cdot \mathbf{n} dS = \iiint_V (\nabla \cdot \mathbf{F}) dV$
    設定:
      - 體積 V: 單位立方體 $[0,1] \times [0,1] \times [0,1]$
      - 向量場: $\mathbf{F} = [x^2, y^2, z^2]$
    """
    # 1. 定義向量場
    F = sp.Matrix([x**2, y**2, z**2])
    
    # 2. 計算 RHS: 體積分 $\iiint (\nabla \cdot \mathbf{F}) dV$
    # div F = 2x + 2y + 2z
    div_F = divergence(F)
    
    # 三重積分
    rhs_value = sp.integrate(div_F, (x, 0, 1), (y, 0, 1), (z, 0, 1))
    # int(2x)dx = 1, *1*1 = 1. 同理 y 和 z. 總和應為 3.
    
    # 3. 計算 LHS: 封閉曲面通量 (6 個面)
    # 由於沒有 surface_integral 函式，我們手動計算每個面的通量 F dot n
    
    # Face x=1 (n=[1,0,0]): F=[1, y^2, z^2]. dot n = 1.
    flux_x1 = sp.integrate(1, (y, 0, 1), (z, 0, 1))
    
    # Face x=0 (n=[-1,0,0]): F=[0, y^2, z^2]. dot n = 0.
    flux_x0 = sp.integrate(0, (y, 0, 1), (z, 0, 1))
    
    # Face y=1 (n=[0,1,0]): F=[x^2, 1, z^2]. dot n = 1.
    flux_y1 = sp.integrate(1, (x, 0, 1), (z, 0, 1))
    
    # Face y=0 (n=[0,-1,0]): F=[x^2, 0, z^2]. dot n = 0.
    flux_y0 = sp.integrate(0, (x, 0, 1), (z, 0, 1))

    # Face z=1 (n=[0,0,1]): F=[x^2, y^2, 1]. dot n = 1.
    flux_z1 = sp.integrate(1, (x, 0, 1), (y, 0, 1))
    
    # Face z=0 (n=[0,0,-1]): F=[x^2, y^2, 0]. dot n = 0.
    flux_z0 = sp.integrate(0, (x, 0, 1), (y, 0, 1))
    
    lhs_value = flux_x1 + flux_x0 + flux_y1 + flux_y0 + flux_z1 + flux_z0
    
    # 4. 驗證
    assert lhs_value == rhs_value, \
        f"散度定理驗證失敗: LHS(Flux)={lhs_value}, RHS(Volume)={rhs_value}"

def test_curl_of_gradient_is_zero():
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


def test_divergence_of_curl_is_zero():
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


def test_line_integral_gradient_theorem():
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
