# dgeom: 微分幾何 $\to$ 相對論

$$\int_{M} d\omega = \int_{\partial M} \omega$$

$$G_{\mu\nu} + \Lambda g_{\mu\nu} = \frac{8\pi G}{c^4} T_{\mu\nu}$$

> 本專案由 [陳鍾誠](https://github.com/ccckmit) 與 Gemini 3 (Pro+Flash) 協作撰寫。

## 書籍

* [AI 電子書：向量微積分+微分幾何+相對論 (第一版)](https://gemini.google.com/share/d13c1e93468b)
* [AI 電子書：向量微積分+微分幾何+相對論 (第二版)](doc/book_v2.md)
    * 第一部分：向量微積分
    * [第 1 章：向量場與純量場 (Vector and Scalar Fields)](https://gemini.google.com/share/0a63a7f9080f)
    * [第 2 章：向量微分運算子 (Vector Differential Operators)](https://gemini.google.com/share/2d6251fbd9dd)
    * [第 3 章：向量積分定理 (Vector Integral Theorems)](https://gemini.google.com/share/82fb86743540)
    * 第二部分：微分幾何
    * [第 4 章：曲線與曲面的基礎 (Fundamentals of Curves and Surfaces)](https://gemini.google.com/share/a24bd9a52fcb)
    * [第 5 章：流形與張量 (Manifolds and Tensors)](https://gemini.google.com/share/6d2b62fc5bd1)
    * [第 6 章：彎曲時空幾何 (Geometry of Curved Spacetime)](https://gemini.google.com/share/4595f6614491)
    * 第三部分：相對論
    * [第 7 章：狹義相對論的原理 (The Principles of Special Relativity)](https://gemini.google.com/share/f305a4d555bb)
    * [第 8 章：閔可夫斯基時空 (Minkowski Spacetime)](https://gemini.google.com/share/2ca1e0f4eaee)
    * [第 9 章：相對論性動力學 (Relativistic Dynamics)](https://gemini.google.com/share/31e416511eac)
    * [第 10 章：等效原理與重力的幾何 (The Equivalence Principle and Geometry of Gravity)](https://gemini.google.com/share/4fe48726a8f7)
    * [第 11 章：愛因斯坦場方程式 (Einstein Field Equations, EFE)](https://gemini.google.com/share/9a1dc5850b8c)
    * [第 12 章：愛因斯坦場方程式的解與應用 (Solutions and Applications of EFE)](https://gemini.google.com/share/3fab28b2e5b4)

## 主模組

[v_calculus.py]:dgeom/sym/v_calculus.py
[d_geometry.py]:dgeom/sym/d_geometry.py
[relativity.py]:dgeom/sym/
[metric.py]:dgeom/sym/metric.py
[d_operator.py]:dgeom/sym/d_operator.py
[tensor.py]:dgeom/sym/tensor.py
[hodge.py]:dgeom/sym/hodge.py
[test_01_v_calculus.py]:test/test_01_v_calculus.py
[test_02_d_geometry.py]:test/test_02_d_geometry.py
[test_02a_tensor.py]:test/test_02a_tensor.py
[test_02b_metric.py]:test/test_02b_metric.py
[test_02c_d_operator.py]:test/test_02c_d_operator.py
[test_03_relativity.py]:test/test_03_relativity.py

主題 | 程式模組 | 測試範例
----|---------|-------
1-向量微積分 | [v_calculus.py] | [test_01_v_calculus.py]
2-微分幾何 |  [d_geometry.py] | [test_02_d_geometry.py]
3-相對論 | [relativity.py] | [test_03_relativity.py]

## 子模組

主題 | 程式模組 | 測試範例
----|---------|-------
張量 | [tensor.py] | [test_02a_tensor.py]
度規 | [metric.py] | [test_02b_metric.py]
外微分 | [d_operator.py] | [test_02c_d_operator.py]
霍奇星算子 | [hodge.py] | [test_02d_hodge.py]
李括號 | [lie_bracket.py] | [test_02e_lie_bracket.py]

## 背後的數學觀念

請參考 [math.md](math.md) !

## 📝 License

MIT License
