# dgeom: 微分幾何 $\to$ 相對論

$$\int_{M} d\omega = \int_{\partial M} \omega$$

$$G_{\mu\nu} + \Lambda g_{\mu\nu} = \frac{8\pi G}{c^4} T_{\mu\nu}$$


> 本專案由 [ccckmit](https://github.com/ccckmit) 與 Gemini 3 (Pro+Flash) 協作撰寫。

## 書籍：從微分幾何到廣義相對論

> -- 以 Python 用 Gemini 實作 dgeom 套件學習

* [00-前言](00-前言.md)
* [01-向量微積分](01-向量微積分.md)
* [02-微分幾何](02-微分幾何.md)
* [03-狹義相對論](03-狹義相對論.md)
* [04-廣義相對論](04-廣義相對論.md)

## 套件：dgeom

程式模組 | 原理 | 說明
-------|-------|----------
[vcalculus.py](dgeom/sym/dvector.py) | [向量微積分] |  梯度 / 散度 / 旋度 / 線積分
[d_operator.py](dgeom/sym/dvector.py) | [外微分] |  $d(d(f)) = 0$
[gstoke.py](dgeom/sym/gstoke.py) | [廣義史托克定理] | $\int_{M} d\omega = \int_{\partial M} \omega$
[riemann.py](dgeom/sym/riemann.py) | [黎曼幾何] | 實作 `metric_tensor`, `christoffel`, `riemann_tensor`, `ricci_tensor` , `ricci_scalar`。
[relativity.py](dgeom/sym/relativity.py) | [相對論]() | 愛因斯坦張量 `einstein_tensor` 。

## 範例


詳細原理說明 | 程式 
-----|------------
[狹義相對論-閩可夫斯基空間](tests/test_minkowski.md) | [test_minkowski.py](tests/test_minkowski.py) | 閩可夫斯基空間
[水星進動-修正軌道誤差](tests/test_murcury_procession.md) | [test_murcury_procession.py](tests/test_murcury_procession.py)
[黑洞-史瓦希度規](tests/test_black_hole.md) | [test_black_hole.py](tests/test_black_hole.py)
[FLRW-均勻且各向同性的宇宙](tests/test_flrw_cosmology.md) | [test_flrw_cosmology.py](tests/test_flrw_cosmology.py)
[SdS 度規](tests/test_schwarzschild_de_sitter.md) |  [test_schwarzschild_de_sitter.py](tests/test_schwarzschild_de_sitter.py)
[微分幾何-黎曼度規](tests/test_riemann.md) | [test_riemann.py](tests/test_riemann.py)
[外微分算子](tests/test_d_operator.md) | [test_d_operator.py](tests/test_d_operator.py)
[廣義史托克定理](tests/test_stoke.md) | [test_stoke.py](tests/test_stoke.py) 

## 📝 License

MIT License