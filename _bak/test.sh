# dgeom.sym 版的微分幾何測試
python -m tests.test_riemann # 黎曼曲率張量測試

# dgeom.sym 版的相對論案例測試
python -m tests.test_minkowski # 閔可夫斯基空間(狹義相對論)
python -m tests.test_schwarzschild_de_sitter # 史瓦西-德西特度規
python -m tests.test_flrw_cosmology # FLRW 宇宙學模型
python -m tests.test_mercury_precession # 水星近日點進動
python -m tests.test_black_hole # 黑洞度規測試

# dgeom.sym 版的向量微積分測試
python -m tests.simple_vcalculus
python -m tests.test_line_integral
python -m tests.test_dvcalculus
python -m tests.test_dvector
python -m tests.test_hodge

# dgeom.num 版的向量微積分測試
python -m tests.test_num_dvector
