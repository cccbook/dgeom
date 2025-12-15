# dgeom 背後的數學

## 數學的程式實作

概念 | 符號公式 | 函數 | 模組
----|------|----------|------------
[梯度] | $\nabla f(\mathbf{x})$ | gradient | [v_calculus.py]
[散度] | $\nabla \cdot \mathbf{F}$ | divergence | [v_calculus.py]
[旋度] | $\nabla \times \mathbf{F}$ | curl | [v_calculus.py]
[線積分] | $\int_C f(x, y, z) \, ds$ | line_integral | [v_calculus.py]
[外微分] | $d\omega = \sum_I df_I \wedge dx_I$ | d_operator | [d_operator.py]
[霍奇星] | $\star: \Omega^k(M) \to \Omega^{n-k}(M)$ | HodgeMetric.star | [hodge.py]
[幾何張量] |$T^{\mu_1\cdots\mu_k}{}_{\nu_1\cdots\nu_l}$  | GeometricTensor | [tensor.py]
[度規張量] | $g_p(\mathbf{u}, \mathbf{v}) \in \mathbb{R}$ | MetricTensor | [metric.py]
[相對論時空] | $G_{\mu\nu} = R_{\mu\nu} - \frac{1}{2} R g_{\mu\nu}$ | Spacetime | [relativity.py]

## 向量微積分定理驗證

[向量微積分] 的測試都在 [test_01_v_calculus.py] 中

概念 | 符號公式 | 函數
----|------|----------
[格林定理] | $\oint_C (P dx + Q dy) = \iint_D \left( \frac{\partial Q}{\partial x} - \frac{\partial P}{\partial y} \right) dA$ | test_green_theorem()
[史托克旋度定理] | $\oint_{\partial S} \mathbf{F} \cdot d\mathbf{r} = \iint_S (\nabla \times \mathbf{F}) \cdot \mathbf{n} \, dS$ | test_stoke_theorem()
[高斯散度定理] | $\iint_S \mathbf{F} \cdot \mathbf{n} \, dS = \iiint_V (\nabla \cdot \mathbf{F}) \, dV$ | test_div_theorem()

## 微分幾何定理驗證

[向量微積分] 的測試都在 [test_02_d_geometry.py] 中

概念 | 符號公式 | 函數 | 測試模組
----|------|----------|------------
[龐加萊引理] | $d(d\omega) = 0$ | test_ddf_is_zero()
[廣義史托克定理] | $\int_{M} d\omega = \int_{\partial M} \omega$ | test_g_stoke_theorem()

## 相對論物理法則驗證

相對論的測試都在 [test_03_relativity.py] 中

[閔可夫斯基空間]:https://gemini.google.com/share/98e0f9a0df9f
[狹義相對論]:https://gemini.google.com/share/7aa859f58771
[重力場方程式]:https://gemini.google.com/share/a1d91f1a442f
[史瓦西黑洞]:https://gemini.google.com/share/7f4edd0b14bf
[水星進動]:https://gemini.google.com/share/1354a9f88c99
[FLRW宇宙膨脹]:https://gemini.google.com/share/ae8faad15c5f
[Kerr旋轉黑洞]:https://gemini.google.com/share/10ebe28fe0ee

概念 | 符號公式 | 函數
----|------|----------
[閔可夫斯基空間] (時空度規) | $ds^2 = c^2 dt^2 - dx^2 - dy^2 - dz^2$ | test_minkowski_ds()
[閔可夫斯基空間] (尺縮:長度收縮) | $L = L_0 \sqrt{1 - \frac{v^2}{c^2}}$ | test_length_contraction()
[閔可夫斯基空間] (鐘慢:時間膨脹) | $d\tau = dt \sqrt{1 - \frac{v^2}{c^2}}$ | test_time_dilation()
[閔可夫斯基空間] (平坦性) | $G_{\mu\nu} = 0$ | test_minkowski_flat()
[狹義相對論] (雙生子佯謬) | $t_B = \frac{2L}{v} \sqrt{1 - \frac{v^2}{c^2}}$ | test_twin_paradox()
[重力場方程式] (左右相等) | $G_{\mu\nu} + \Lambda g_{\mu\nu} = \frac{8\pi G}{c^4} T_{\mu\nu}$ | test_einstein_field_equation()
[史瓦西黑洞] (真空性) | $G_{\mu\nu}=0$ | test_schwarzschild_vacuum()
[史瓦西黑洞] (史瓦西半徑) | $R_s = \frac{2 G M}{c^2}$ | 
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
[霍奇星]:https://gemini.google.com/share/867983c498e6
[梯度]:https://gemini.google.com/share/7a3b689e32b0
[散度]:https://gemini.google.com/share/8e63457e5dca
[旋度]:https://gemini.google.com/share/4c454d319204
[線積分]:https://gemini.google.com/share/a372b1ed96ee
[黎曼度規]:https://gemini.google.com/share/c094e1f36905
[龐加萊引理]:https://gemini.google.com/share/1073261c1e39
[張量]:https://gemini.google.com/share/80764d8ab893

[幾何張量]:https://gemini.google.com/share/3cf638068d2e
[度規張量]:https://gemini.google.com/share/844b156e3149
[相對論時空]:https://gemini.google.com/share/43e3d66179e0

[格林定理]:https://gemini.google.com/share/1a1e89bb4bbf
[史托克旋度定理]:https://gemini.google.com/share/199c4917addc
[高斯散度定理]:https://gemini.google.com/share/6c7252352c3e



[metric.py]:dgeom/sym/metric.py
[d_operator.py]:dgeom/sym/d_operator.py
[tensor.py]:dgeom/sym/tensor.py
[hodge.py]:dgeom/sym/hodge.py
[manifold.py]:dgeom/sym/manifold.py
[test_02a_tensor.py]:test/test_02a_tensor.py
[test_02b_metric.py]:test/test_02b_metric.py
[test_02c_d_operator.py]:test/test_02c_d_operator.py
[test_02d_hodge.py]:test/test_02c_hodge.py
[test_02e_manifold.py]:test/test_02e_manifold.py

[v_calculus.py]:dgeom/sym/v_calculus.py
[d_geometry.py]:dgeom/sym/d_geometry.py
[relativity.py]:dgeom/sym/

[test_01_v_calculus.py]:test/test_01_v_calculus.py
[test_02_d_geometry.py]:test/test_02_d_geometry.py
[test_03_relativity.py]:test/test_03_relativity.py
