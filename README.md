# dgeom: 微分幾何 $\to$ 相對論

$$\int_{M} d\omega = \int_{\partial M} \omega$$

$$G_{\mu\nu} + \Lambda g_{\mu\nu} = \frac{8\pi G}{c^4} T_{\mu\nu}$$

> 本專案由 [陳鍾誠](https://github.com/ccckmit) 與 Gemini 3 (Pro+Flash) 協作撰寫。

## 書籍

* [AI 電子書：向量微積分+微分幾何+相對論](https://gemini.google.com/share/d13c1e93468b)

## 套件：dgeom

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

主題 | 程式模組 | 測試範例
----|---------|-------
[向量微積分] | [vcalculus.py](dgeom/sym/vcalculus.py) | [test_01_vcalculus.py](test/test_01_vcalculus.py)
[微分幾何] |  [dgeometry.py](dgeom/sym/dgeometry.py) | [test_02_dgeometry.py](test/test_02_dgeometry.py)
[相對論] | [relativity.py](dgeom/sym/relativity.py) | [test_03_special_relativity.py](test/test_03_special_relativity.py) <br/> [test_04_general_relativity.py](test/test_04_general_relativity.py) 

## 數學

概念 | 符號公式 | 實作
----|------|----------
[梯度] | $\nabla f(\mathbf{x})$ | gradient
[散度] | $\nabla \cdot \mathbf{F}$ | divergence
[旋度] | $\nabla \times \mathbf{F}$ | curl
[線積分] | $\int_C f(x, y, z) \, ds$ | line_integral
[外微分] | $d\omega = \sum_I df_I \wedge dx_I$ | d_operator
[龐加萊引理] | $$d(d\omega) = 0$$
[廣義史托克定理] | $\int_{M} d\omega = \int_{\partial M} \omega$ | 
[黎曼度規] | $g_{ij}(p) = g_p\left(\frac{\partial}{\partial x^i}, \frac{\partial}{\partial x^j}\right)$ | class Metrics
[狹義相對論] | $ds^2 = -c^2 dt^2 + dx^2 + dy^2 + dz^2$  | minkowski_metric()
[廣義相對論] | $G_{\mu\nu} + \Lambda g_{\mu\nu} = \frac{8\pi G}{c^4} T_{\mu\nu}$ | class RelativityMetrics

## 📝 License

MIT License