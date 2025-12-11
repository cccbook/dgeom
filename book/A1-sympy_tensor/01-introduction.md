# 第一章：前言與環境準備 (Introduction & Setup)

在開始推導廣義相對論或連續體力學公式之前，我們必須先理解 SymPy 是如何「看待」張量的。與數值計算庫（如 NumPy 或 PyTorch）不同，SymPy 的強項在於**符號推導**與**抽象運算**。

## 1.1 SymPy 張量模組簡介

SymPy 的張量功能其實分為兩個主要流派，初學者容易混淆。在學習之前，請務必區分這兩者：

### 1. 陣列運算 (N-dim Arrays)
- **模組位置**：`sympy.tensor.array`
- **概念**：這是 NumPy 的符號版。它處理的是**具體的**分量。
- **適用場景**：當你已經知道張量的維度（例如 3x3 矩陣）以及每個位置的函數（如 $x^2, \sin(y)$），需要進行微分或外積時。
- **例子**：
  $$ A = \begin{bmatrix} x & y \\ z & 0 \end{bmatrix} $$

### 2. 抽象張量演算 (Abstract Tensor Calculus)
- **模組位置**：`sympy.tensor.tensor`
- **概念**：這是物理學家熟悉的 **Penrose 抽象指標記法 (Abstract Index Notation)**。
- **適用場景**：當你不需要知道維度是多少，也不關心具體分量，只想推導公式本身時。這是廣義相對論推導的核心。
- **例子**：
  $$ R_{\mu\nu} - \frac{1}{2}g_{\mu\nu}R + g_{\mu\nu}\Lambda = \frac{8\pi G}{c^4} T_{\mu\nu} $$
  *（在這裡，我們不關心 $\mu$ 是 0 還是 1，我們關心的是代數結構。）*

> **本教材的核心**：我們將重點放在 **`sympy.tensor.tensor` (抽象張量)**，因為這是 SymPy 最獨特且強大的部分，最後再輔以陣列運算來代入數值。

---

## 1.2 環境安裝與基礎配置

為了獲得最佳的學習體驗，強烈建議使用 **Jupyter Notebook** 或 **JupyterLab**，因為張量公式如果只用純文字顯示（如 `A(mu, nu)`）會非常難以閱讀，而 Jupyter 可以渲染出漂亮的 $\LaTeX$ 數學格式。

### 步驟 1：安裝

如果您尚未安裝 SymPy，請在終端機（Terminal）或命令提示字元中執行：

```bash
pip install sympy notebook
```

### 步驟 2：標準引入慣例 (Boilerplate Code)

在你的 Python 腳本或 Notebook 的開頭，使用以下設定。這是我們接下來所有章節的標準起手式：

```python
import sympy
from sympy import symbols, init_printing

# 引入抽象張量相關的核心類別
from sympy.tensor.tensor import TensorIndexType, TensorIndex, TensorHead, tensor_indices

# 啟用漂亮的數學顯示 (Pretty Printing)
# use_latex='mathjax' 確保在 Jupyter 中渲染出完美的數學公式
init_printing(use_latex='mathjax')

print(f"SymPy Version: {sympy.__version__}")
```

> SymPy Version: 1.14.0

### 步驟 3：Hello Tensor World (測試環境)

讓我們寫一段簡單的程式碼，測試環境是否設定成功。我們將定義一個抽象的空間，並創建一個帶有上指標的向量 $A^\mu$。

```python
# 1. 定義一個指標類型 (Index Type)
# 例如：Lorentz 空間 (L)，通常用於相對論
Lorentz = TensorIndexType('Lorentz', dummy_name='L')

# 2. 定義具體的指標符號 (Indices)
# mu, nu 是這個空間中的指標
mu, nu = tensor_indices('mu nu', Lorentz)

# 3. 定義一個張量實體 (Tensor Head)
# A 是一個張量，且我們指定它通常帶有一個指標
A = TensorHead('A', [Lorentz])

# 4. 建立張量表達式
# 創建一個表達式：A 的 mu 分量
expr = A(mu)

# 5. 顯示結果
print("Text Output:", expr)
# 在 Jupyter Notebook 中，直接輸入變數名稱會顯示 LaTeX 格式
expr
```

**預期輸出結果：**

如果您的環境設定正確，`expr` 在 Jupyter Notebook 中應該會顯示為漂亮的數學符號：
$$ A^\mu $$

而在 `print` 純文字輸出中，可能會顯示類似：
`A(mu)`

---

### 常見問題 (FAQ)

**Q: 為什麼不用 `from sympy import *`？**
A: 雖然方便，但 SymPy 的命名空間非常大，容易與其他庫（如 NumPy）衝突。此外，明確導入 `TensorIndex` 等類別，能幫助你記住哪些函數屬於張量模組。

**Q: 輸出的公式看起來像亂碼或 ASCII 藝術？**
A: 請確認你有執行 `init_printing()`。如果你不在 Jupyter 環境中（例如使用 PyCharm 或 VS Code 的終端機），SymPy 會嘗試用 Unicode 字符拼湊出數學式，雖然不完美但仍可讀。
