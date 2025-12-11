## 📚 SymPy 張量 (Tensor) 處理教材目錄

### 第一章：張量基礎概念與 SymPy 環境建立

1.  **什麼是張量？**
    * 從純量（Scalar）到向量（Vector）再到張量
    * 張量的階（Rank）與分量（Components）
    * 協變（Covariant）與逆變（Contravariant）指標的概念
2.  **SymPy 環境設定**
    * 安裝與導入必要的模組
    * 張量運算的基礎設定
3.  **定義流形（Manifold）與座標系統**
    * `Manifold` 與 `CoordSysCartesian` 的使用

---

### 第二章：SymPy 中張量的定義與基本操作

1.  **定義張量場（Tensor Field）**
    * 使用 `TensorField` 類建立張量
    * 指定張量的階與指標類型
    * $T_{ij}$ 和 $T^{ij}$ 的表示法
2.  **張量分量的存取與操作**
    * 取得張量在特定座標下的分量
    * 使用 `get_components()` 方法
3.  **基本張量運算**
    * 張量的加法與減法
    * 張量的純量乘法

---

### 第三章：張量積與縮約（Contraction）

1.  **外積（Outer Product）**
    * 定義兩個張量的外積
    * 外積對張量階數的影響
2.  **內積（Inner Product）與縮約**
    * 縮約的定義與數學表示式：例如，若 $\mathbf{A}$ 的指標是 $(i, j)$，$\mathbf{B}$ 的指標是 $(k, l)$，則縮約後的指標表示為：
        $$C_{kl} = \sum_{j} A_{ij} B^{jk}$$
    * 使用 `tensorcontraction` 函式進行縮約
    * 克羅內克 $\delta$（Kronecker delta）張量
3.  **度規張量（Metric Tensor）**
    * 定義與使用度規張量 $g_{ij}$
    * 升降指標（Raising and Lowering Indices）的操作

---

### 第四章：張量的微分運算

1.  **協變微分（Covariant Differentiation）**
    * Christoffel 符號（Christoffel Symbols）的計算
    * 協變導數的定義與計算
    * 向量場的協變導數：
        $$\nabla_j V^i = \partial_j V^i + \Gamma^i_{jk} V^k$$
        其中 $\Gamma^i_{jk}$ 是 Christoffel 符號。
2.  **黎曼曲率張量（Riemann Curvature Tensor）**
    * 定義與計算公式
    * Ricci 張量（Ricci Tensor）與純量曲率（Scalar Curvature）

---

### 第五章：特殊張量與應用

1.  **排列張量（Permutation Tensor）**
    * Levi-Civita 符號 $\epsilon_{ijk}$ 的使用
2.  **能量動量張量（Stress-Energy Tensor）** (簡要介紹)
3.  **SymPy 張量在特定座標系統下的應用**
    * 極座標、球座標等非笛卡爾座標系的運算範例

