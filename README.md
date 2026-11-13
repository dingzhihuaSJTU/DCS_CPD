# 复现代码说明

本目录包含论文全部数值结果的生成脚本与结果文件。所有脚本只依赖 numpy / scipy / pandas / matplotlib / ruptures，无 GPU、无网络、无随机外部数据。

## 文件职责

| 文件 | 职责 |
|---|---|
| `dcs_cpd.py` | DCS-CPD 参考实现。函数与论文公式一一对应：`cosine_series`↔式(4)，`threshold`↔式(12)，`score_constants`/`critical_count`↔式(25)(26)(33)，`score_curve`↔式(13)(34)，`localize`↔算法 2 |
| `baselines.py` | 18 个基线。ruptures 的 12 个（4 方法 × 3 代价），加 MOSUM、移动相关、CUSUM、BOCPD |
| `metrics.py` | 一对一匹配的 precision/recall/F1 与定位误差 |
| `synthetic.py` | 四族突变生成器：level / polarity / phase / period，噪声方差按 SNR 定义设置 |
| `run_synthetic.py` | 主实验，输出 `results/synthetic.csv` |
| `run_real.py` | 真实数据实验（TCPD 六条 + HASC），输出 `results/real.csv` |
| `run_support.py` | 五个子命令：`runtime` / `sensitivity` / `null` / `calibration` / `period` |
| `run_approx.py` | 零分布矩与尾概率的预测/实测对照，输出 `results/approx.csv` |
| `make_figures.py` | 生成 `figs/score_example.pdf`、`period_sweep.pdf`、`runtime.pdf` |

## 结果文件与论文表格的对应

| 论文位置 | 结果文件 | 生成命令 |
|---|---|---|
| 表 4（合成主表） | `results/synthetic.csv` | `run_synthetic.py --reps 15` |
| 表 5（真实数据） | `results/real.csv` | `run_real.py` |
| 表 2（近似矩） | `results/approx.csv` | `run_approx.py` |
| 表 3（尾概率） | `results/approx.csv` | `run_approx.py` |
| 表 6（虚警） | `results/null.csv` | `run_support.py null --reps 6` |
| 表 1（敏感性） | `results/sensitivity.csv` | `run_support.py sensitivity --reps 10` |
| 图 5（分数示例） | 运行时生成 | `make_figures.py` |
| 图 6（周期扫描） | `results/period.csv` | `run_support.py period --reps 8` |
| 图 7（运行时间） | `results/runtime.csv` | `run_support.py runtime --reps 2` |
| 附录表 B.8（18 方法全表） | `results/synthetic.csv` | 同上 |

