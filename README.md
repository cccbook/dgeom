# dgeom: 微分幾何 $\to$ 相對論

$$\int_{M} d\omega = \int_{\partial M} \omega$$

$$G_{\mu\nu} + \Lambda g_{\mu\nu} = \frac{8\pi G}{c^4} T_{\mu\nu}$$

> 本專案由 [陳鍾誠](https://github.com/ccckmit) 與 Gemini 3 (Pro+Flash) 協作撰寫。

## 書籍

* [AI 電子書：向量微積分+微分幾何+相對論](https://gemini.google.com/share/d13c1e93468b)
    * [第 1 章：向量場與純量場 (Vector and Scalar Fields)](https://gemini.google.com/share/0a63a7f9080f)
    * [第 2 章：向量微分運算子 (Vector Differential Operators)](https://gemini.google.com/share/2d6251fbd9dd)
    * [第 3 章：向量積分定理 (Vector Integral Theorems)](https://gemini.google.com/share/82fb86743540)
    * [第 4 章：曲線與曲面的基礎 (Fundamentals of Curves and Surfaces)](https://gemini.google.com/share/a24bd9a52fcb)
    * [第 5 章：流形與張量 (Manifolds and Tensors)](https://gemini.google.com/share/6d2b62fc5bd1)
    * [第 6 章：彎曲時空幾何 (Geometry of Curved Spacetime)](https://gemini.google.com/share/4595f6614491)
    * [第 7 章：狹義相對論的原理 (The Principles of Special Relativity)](https://gemini.google.com/share/f305a4d555bb)
    * [第 8 章：閔可夫斯基時空 (Minkowski Spacetime)](https://gemini.google.com/share/2ca1e0f4eaee)
    * [第 9 章：相對論性動力學 (Relativistic Dynamics)](https://gemini.google.com/share/31e416511eac)
    * [第 10 章：等效原理與重力的幾何 (The Equivalence Principle and Geometry of Gravity)](https://gemini.google.com/share/4fe48726a8f7)
    * [第 11 章：愛因斯坦場方程式 (Einstein Field Equations, EFE)](https://gemini.google.com/share/9a1dc5850b8c)
    * [第 12 章：愛因斯坦場方程式的解與應用 (Solutions and Applications of EFE)](https://gemini.google.com/share/3fab28b2e5b4)

## 主要模組



主題 | 程式模組 | 測試範例
----|---------|-------
[向量微積分] | [vcalculus.py](dgeom/sym/vcalculus.py) | [test_01_vcalculus.py](test/test_01_vcalculus.py)
[微分幾何] |  [dgeometry.py](dgeom/sym/dgeometry.py) | [test_02_dgeometry.py](test/test_02_dgeometry.py)
[相對論] | [relativity.py](dgeom/sym/relativity.py) | [test_03_special_relativity.py](test/test_03_special_relativity.py) <br/> [test_04_general_relativity.py](test/test_04_general_relativity.py) 

## 重要物件

概念 | 類別 | 符號公式 | 說明
-----|----|------|----------
流形 | [class Manifold](https://gemini.google.com/share/4cd49f6f253f) | 高維可微分曲面
張量 | [class GeometricTensor](https://gemini.google.com/share/3cf638068d2e) | $T^{\mu_1\cdots\mu_k}{}_{\nu_1\cdots\nu_l}$ | 座標轉換的函數 (用『高維陣列』表示)
度規張量 | [class MetricTensor(GeometricTensor)](https://gemini.google.com/share/844b156e3149) | $g_p(\mathbf{u}, \mathbf{v}) \in \mathbb{R}$ | 對稱正定的二階協變張量，用來測量（長度、角度、體積...）
相對論時空 | [class Spacetime](https://gemini.google.com/share/43e3d66179e0) | $G_{\mu\nu} = R_{\mu\nu} - \frac{1}{2} R g_{\mu\nu}$ | 愛因斯坦重力場方程式

<!--
[class Form]() |
[class TangentVector]() |
[class ParametrizedDomain]() |
[class HyperCube(ParametrizedDomain)]() |
[class ParametricPatch(HyperCube)]() |
[class TangentVector]() |
-->

## 數學

概念 | 符號公式 | 實作
----|------|----------
[梯度] | $\nabla f(\mathbf{x})$ | gradient
[散度] | $\nabla \cdot \mathbf{F}$ | divergence
[旋度] | $\nabla \times \mathbf{F}$ | curl
[線積分] | $\int_C f(x, y, z) \, ds$ | line_integral
[外微分] | $d\omega = \sum_I df_I \wedge dx_I$ | d_operator
[龐加萊引理] | $d(d\omega) = 0$ | 
[廣義史托克定理] | $\int_{M} d\omega = \int_{\partial M} \omega$ | 
[張量] | $T^{\mu_1\cdots\mu_k}{}_{\nu_1\cdots\nu_l}$  | [numpy+sympy](https://gemini.google.com/share/012d20119bb9)
[黎曼度規] | $g_{ij}(p) = g_p\left(\frac{\partial}{\partial x^i}, \frac{\partial}{\partial x^j}\right)$ | class Metrics
[狹義相對論] | $ds^2 = -c^2 dt^2 + dx^2 + dy^2 + dz^2$  | minkowski_metric()
[廣義相對論] | $G_{\mu\nu} + \Lambda g_{\mu\nu} = \frac{8\pi G}{c^4} T_{\mu\nu}$ | class RelativityMetrics

## 📝 License

MIT License

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