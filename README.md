# dgeom: Symbolic Differential Geometry & General Relativity

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

## 🌌 物理驗證案例 (Verification Cases)

本專案包含多個高強度的物理測試案例，證明了計算引擎的強健性：

| 測試案例 | 描述 | 驗證目標 |
| :--- | :--- | :--- |
| **Minkowski** | 狹義相對論平坦時空 | 確認所有曲率張量均為 0 |
| **Schwarzschild** | 靜態、不帶電黑洞 | 驗證 $G_{\mu\nu} = 0$ (真空解) |
| **Reissner-Nordström** | 帶電黑洞 | 驗證幾何與電磁能量動量張量的耦合 |
| **Kerr Metric** | **旋轉黑洞** (高難度) | 處理非對角度規與參考系拖曳，驗證 $R_{\mu\nu}=0$ |
| **FLRW Metric** | 宇宙學標準模型 | 推導弗里德曼方程式 (Friedmann Eqs) |
| **Mercury Precession** | 水星近日點進動 | 推導測地線方程式與有效位能修正項 $-3ML^2/r^4$ |

## dgeom 測試案例與數學原理解說

以下使用 dgeom.sym 套件（背後是 sympy）

* [test_minkowski.py](tests/test_minkowski.py) ： 狹義相對論『閩可夫斯基空間』範例
    * 數學原理 -- https://gemini.google.com/share/1f70c33d6a06
* [test_murcury_procession.py](tests/test_murcury_procession.py) ：廣義相對論『水星進動』範例
    * 數學原理 -- https://gemini.google.com/share/9a7c48879a1a
* [test_black_hole.py](tests/test_black_hole.py) ：廣義相對論『黑洞』範例
    * 數學原理 -- https://gemini.google.com/share/529f1fee7e0e
* [test_flrw_cosmology.py](test_flrw_cosmology.py) ： 廣義相對論 FLRW 度規 (均勻且各向同性的宇宙)
    * 數學原理 -- https://gemini.google.com/share/0dfe745d2040
* [test_schwarzschild_de_sitter.py](tests/test_schwarzschild_de_sitter.py) ： 廣義相對論 SdS 度規範例
    * 數學原理 -- https://gemini.google.com/share/10ac77736c53
* [test_riemann.py](tests/test_riemann.py) ：微分幾何範例（度規）
    * 數學原理 -- https://gemini.google.com/share/de0e3b5ee633
* [test_dvector.py](tests/test_dvector.py) : 向量微積分範例（含外微分與廣義史托克定理）(使用 sympy 實作)
    * 數學原理 -- https://gemini.google.com/share/66966e45f718

以下使用 dgeom.num 套件 (背後是 numpy)

* [test_num_dvector.py](tests/test_num_dvector.py) ：向量微積分範例（含外微分與廣義史托克定理）
    * 數學原理 -- https://gemini.google.com/share/cf1526765a9f

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

*   `dgeom/sym/dvector.py`: 實作 `TangentVector`, `Form`, `d` (外微分), `lie_bracket`。
*   `dgeom/sym/riemann.py`: 實作 `metric_tensor`, `christoffel`, `riemann_tensor`, `ricci_tensor` (矩陣化), `ricci_scalar`。
*   `dgeom/sym/relativity.py`: 實作 `einstein_tensor` (矩陣化)。

## 📝 License

MIT License