# dgeom 背後的數學

## 重要物件

[幾何張量]:https://gemini.google.com/share/3cf638068d2e
[度規張量]:https://gemini.google.com/share/844b156e3149
[相對論時空]:https://gemini.google.com/share/43e3d66179e0
[class GeometricTensor]:dgeom/sym/tensor.py
[class MetricTensor(GeometricTensor)]:dgeom/sym/metric.py
[class Spacetime]:dgeom/sym/relativity.py

概念 | 類別 (class) | 模組 | 符號公式
-----|------------|------|-----
[幾何張量] | GeometricTensor | [tensor.py] |$T^{\mu_1\cdots\mu_k}{}_{\nu_1\cdots\nu_l}$ 
[度規張量] | MetricTensor | [metric.py] | $g_p(\mathbf{u}, \mathbf{v}) \in \mathbb{R}$
[相對論時空] | Spacetime | [relativity.py] | $G_{\mu\nu} = R_{\mu\nu} - \frac{1}{2} R g_{\mu\nu}$


## 數學函數實作

概念 | 符號公式 | 函數 | 模組
----|------|----------|------------
[梯度] | $\nabla f(\mathbf{x})$ | gradient | 
[散度] | $\nabla \cdot \mathbf{F}$ | divergence | 
[旋度] | $\nabla \times \mathbf{F}$ | curl | 
[線積分] | $\int_C f(x, y, z) \, ds$ | line_integral | 
[外微分] | $d\omega = \sum_I df_I \wedge dx_I$ | d_operator

## 向量微積分定理驗證

向量微積分的測試都在 [test_01_vcalculus.py] 中

概念 | 符號公式 | 函數
----|------|----------
[格林定理] | $\oint_C (P dx + Q dy) = \iint_D \left( \frac{\partial Q}{\partial x} - \frac{\partial P}{\partial y} \right) dA$ | test_green_theorem()
[史托克旋度定理] | $\oint_{\partial S} \mathbf{F} \cdot d\mathbf{r} = \iint_S (\nabla \times \mathbf{F}) \cdot \mathbf{n} \, dS$ | test_stoke_theorem()
[高斯散度定理] | $\iint_S \mathbf{F} \cdot \mathbf{n} \, dS = \iiint_V (\nabla \cdot \mathbf{F}) \, dV$ | test_div_theorem()

## 微分幾何定理驗證

概念 | 符號公式 | 函數 | 測試模組
----|------|----------|------------
[龐加萊引理] | $d(d\omega) = 0$ | test_ddf_is_zero()
[廣義史托克定理] | $\int_{M} d\omega = \int_{\partial M} \omega$ | test_g_stoke_theorem()

## 狹義相對論物理法則驗證

狹義相對論的測試都在 [test_03_s_relativity.py] 中

概念 | 符號公式 | 函數
----|------|----------
[閔可夫斯基空間:時空度規] | $ds^2 = c^2 dt^2 - dx^2 - dy^2 - dz^2$ | test_minkowski_ds()
[閔可夫斯基空間:尺縮] (長度收縮) | $L = L_0 \sqrt{1 - \frac{v^2}{c^2}}$ | test_length_contraction()
[閔可夫斯基空間:鐘慢] (時間膨脹) | $d\tau = dt \sqrt{1 - \frac{v^2}{c^2}}$ | test_time_dilation()
[閔可夫斯基空間:平坦性] | $G_{\mu\nu} = 0$ | test_minkowski_flat()
[狹義相對論：雙生子佯謬] | $t_B = \frac{2L}{v} \sqrt{1 - \frac{v^2}{c^2}}$ | test_twin_paradox()

## 廣義相對論物理法則驗證

[廣義相對論] 的測試都在 [test_04_g_relativity.py] 中

概念 | 符號公式 | 函數
----|------|----------
[重力場方程式：左右相等] | $G_{\mu\nu} + \Lambda g_{\mu\nu} = \frac{8\pi G}{c^4} T_{\mu\nu}$ | test_einstein_field_equation()
[史瓦西黑洞：真空性] | $G_{\mu\nu}=0$ | test_schwarzschild_vacuum()
[史瓦西黑洞：半徑] | $R_s = \frac{2 G M}{c^2}$ | 
[水星進動] (差43秒角) | $\Delta \phi = \frac{24 \pi^3 a^2}{T^2 c^2 (1 - e^2)}$ | test_metest_mercury_precession()
[FLRW宇宙膨脹] (膨脹＋各向同性) | $d\Omega^2 = d\theta^2 + \sin^2 \theta d\phi^2$ | test_flrw_expansion() | $G_{\phi\phi} = G_{\theta\theta} \sin^2 \theta$
[Kerr旋轉黑洞] (自旋+對稱) | $g_{t,\phi}=g_{\phi,t}$ , $a \in g_{\phi,t}$ | test_kerr_black_hole()


[向量微積分]:https://gemini.google.com/share/696c3b3e23d4
[微分幾何]:https://gemini.google.com/share/5b5da9a9a179
[相對論]:https://gemini.google.com/share/53f1d73cc51c
[狹義相對論]:https://gemini.google.com/share/d8d96751f5b1
[廣義相對論]:https://gemini.google.com/share/23e50017bf00
[廣義史托克定理]:https://gemini.google.com/share/548c3712f2f7
[度規]:https://gemini.google.com/share/ae5f5d47714d
[外微分]:https://gemini.google.com/share/1202f0099ce2
[霍奇星算子]:https://gemini.google.com/share/867983c498e6
[梯度]:https://gemini.google.com/share/7a3b689e32b0
[散度]:https://gemini.google.com/share/8e63457e5dca
[旋度]:https://gemini.google.com/share/4c454d319204
[線積分]:https://gemini.google.com/share/a372b1ed96ee
[黎曼度規]:https://gemini.google.com/share/c094e1f36905
[龐加萊引理]:https://gemini.google.com/share/1073261c1e39
[張量]:https://gemini.google.com/share/80764d8ab893
