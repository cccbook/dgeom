# dgeom: 『微分幾何 => 相對論』的 python 套件

$$\int_{M} d\omega = \int_{\partial M} \omega$$

$$G_{\mu\nu} + \Lambda g_{\mu\nu} = \frac{8\pi G}{c^4} T_{\mu\nu}$$

> 本專案由 [ccckmit](https://github.com/ccckmit)  指揮 Gemini 撰寫

**dgeom** 是一個基於 [SymPy](https://www.sympy.org/) 的輕量級 Python 函式庫，專為微分幾何與廣義相對論的符號運算而設計。

它旨在縮短抽象數學定義（如外微分、流形微積分）與具體物理計算（如愛因斯坦張量、黑洞解）之間的距離。從基礎的 **Stokes 定理** 驗證，到複雜的 **Kerr 旋轉黑洞** 真空解驗證，本專案皆能精確處理。

## ✨ 特色 (Features)

*   **純符號運算**：基於 SymPy，所有計算結果皆為精確的解析解（Analytical Solutions），無數值誤差。
*   **流形微積分 (Calculus on Manifolds)**：
    *   支援坐標無關的運算，如外微分算子 ($d$)、李括號 (Lie Bracket)、切向量場。
    *   實作廣義 Stokes 定理 $\int_{\partial \Omega} \omega = \int_{\Omega} d\omega$。
*   **黎曼幾何 (Riemannian Geometry)**：
    *   自動計算 Christoffel 符號 ($\Gamma^\lambda_{\mu\nu}$)。
    *   黎曼曲率張量 ($R^\rho_{\sigma\mu\nu}$)、Ricci 張量 ($R_{\mu\nu}$) 與 Ricci 純量 ($R$)。
*   **廣義相對論 (General Relativity)**：
    *   支援任意度規的愛因斯坦張量 ($G_{\mu\nu}$) 計算。
    *   驗證經典黑洞解（Schwarzschild, Reissner-Nordström, Kerr）。
    *   支援動態時空（FLRW 宇宙學度規）。

## 📦 安裝 (Installation)

本專案主要依賴 `sympy` (但向量微積分部分也有 numpy 的版本)

```bash
git clone https://github.com/ccc-py/dgeom.git

cd dgeom

pip install sympy numpy

./test.sh
```

## 🚀 快速開始 (Quick Start)

### 1. 計算史瓦西度規的曲率

```python
import sympy as sp
from dgeom.sym import ricci_tensor, ricci_scalar, einstein_tensor

# 定義座標與參數
t, r, theta, phi = sp.symbols('t r theta phi')
coords = [t, r, theta, phi]
rs = sp.symbols('r_s') # 史瓦西半徑

# 定義史瓦西度規 (Covariant)
f = 1 - rs/r
G_cov = sp.diag(-f, 1/f, r**2, r**2 * sp.sin(theta)**2)
G_cont = sp.diag(-1/f, f, 1/r**2, 1/(r**2 * sp.sin(theta)**2))

# 計算愛因斯坦張量
R_mn = ricci_tensor(G_cont, G_cov, coords)
R_scalar = ricci_scalar(R_mn, G_cont)
G_mn = einstein_tensor(R_mn, R_scalar, G_cov)

# 驗證真空解 (應為 0 矩陣)
print("Einstein Tensor:", sp.simplify(G_mn))
```

### 2. 驗證 Stokes 定理 (微分形式)

```python
from dgeom.sym import Form, d, integrate_form, ParametricPatch
# ... (定義 Form 與 Domain)
# 驗證 ∫ d(omega) = ∫ omega 在邊界
```

## dgeom 測試案例與數學原理解說

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

## 🧪 執行測試 (Running Tests)

全部測試

```bash
./test.sh
```

單獨測試

```bash
# dgeom.sym 版的微分幾何測試
python -m tests.test_riemann # 黎曼曲率張量測試

# dgeom.sym 版的相對論案例測試
python -m tests.test_minkowski # 閔可夫斯基空間(狹義相對論)
python -m tests.test_schwarzschild_de_sitter # 史瓦西-德西特度規
python -m tests.test_flrw_cosmology # FLRW 宇宙學模型
python -m tests.test_mercury_precession # 水星近日點進動
python -m tests.test_black_hole # 黑洞度規測試

# dgeom.sym 版的向量微積分測試
python -m tests.test_dvector

# dgeom.num 版的向量微積分測試
python -m tests.test_num_dvector
```

## 📂 專案結構 (Project Structure)

程式模組 | 原理 | 說明
-------|-------|----------
[d_operator.py](dgeom/sym/dvector.py) | [外微分]() |  $d(d(f)) = 0$ 
[gstoke.py](dgeom/sym/dvector.py) | [積分：廣義史托克定理]() | $\int_{M} d\omega = \int_{\partial M} \omega$
[riemann.py](dgeom/sym/riemann.py) | [黎曼幾何]() | 實作 `metric_tensor`, `christoffel`, `riemann_tensor`, `ricci_tensor` , `ricci_scalar`。
[relativity.py](dgeom/sym/relativity.py) | [相對論]() | 實作 `einstein_tensor` 。

## 📝 License

MIT License