# dgeom: 微分幾何 $\to$ 相對論

$$\int_{M} d\omega = \int_{\partial M} \omega$$

$$G_{\mu\nu} + \Lambda g_{\mu\nu} = \frac{8\pi G}{c^4} T_{\mu\nu}$$

> 本專案由 [陳鍾誠](https://github.com/ccckmit) 與 Gemini 3 (Pro+Flash) 協作撰寫。

## 書籍

* [AI 電子書：向量微積分+微分幾何+相對論](https://gemini.google.com/share/d13c1e93468b) -- (作者：陳鍾誠+Gemini 3 Flash)
    * [提示詞](book/00.a-提示詞.md)
    * [寫作動機](book/00.b-前言.md)
    * [專有名詞索引](book/00.c-專有名詞索引.md)

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

主題 | 程式模組 | 概念 | 數學
----|---------|--------|----
[向量微積分] | [vcalculus.py](dgeom/sym/vcalculus.py) |  [梯度] / [散度] / [旋度] / [線積分] | $\nabla f(\mathbf{x}) = \left\langle \frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \ldots, \frac{\partial f}{\partial x_n} \right\rangle$ $\nabla \cdot \mathbf{F} = \frac{\partial P}{\partial x} + \frac{\partial Q}{\partial y} + \frac{\partial R}{\partial z}$ $\nabla \times \mathbf{F} = \begin{vmatrix} \mathbf{i} & \mathbf{j} & \mathbf{k} \\ \frac{\partial}{\partial x} & \frac{\partial}{\partial y} & \frac{\partial}{\partial z} \\ P & Q & R \end{vmatrix}$
[微分幾何] |  [dgeometry.py](dgeom/sym/dgeometry.py) | [外微分] / [霍奇星算子] / [廣義史托克定理] / [黎曼度規] |   $d\omega = \sum_I df_I \wedge dx_I$ $\int_{M} d\omega = \int_{\partial M} \omega$ $g_{ij}(p) = g_p\left(\frac{\partial}{\partial x^i}, \frac{\partial}{\partial x^j}\right)$
[相對論] | [relativity.py](dgeom/sym/relativity.py) | [狹義相對論] / [廣義相對論] | $ds^2 = c^2 dt^2 - (dx)^2 - (dy)^2 - (dz)^2$ $G_{\mu\nu} + \Lambda g_{\mu\nu} = \frac{8\pi G}{c^4} T_{\mu\nu}$

## 📝 License

MIT License