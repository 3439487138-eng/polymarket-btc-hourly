# Polymarket × Binance BTC 小时盘研究

在每个 BTC 小时盘开始后第 30 分钟，预测该场最终结算为 Up 的概率。以 Polymarket 最近有效成交隐含的概率为基准，检验 Binance 现货与 USDT 永续行情是否提供额外信息。

[研究总报告（PDF）](reports/research_summary.pdf) · [数据字典](data/README.md) · [时间审计](reports/appendix/timing_audit.md)

## 数据和时间划分

- 北京时间 2025-05-28 13:00 至 2025-12-31 21:00。预选并完成查询 5,176 场，保存 7,832,927 条窗口内成交，5,140 场可直接建模。每场均对齐 Binance 现货和永续各 60 根分钟线。期间有 41 个小时未纳入样本，另有 36 个已采集场次因指定区间缺少有效 Polymarket 成交而未入模。
- 5–11 月共 4,410 场可用。8–11 月按月滚动验证，共 2,919 场，用于选择超参数。12 月用 730 场复核。
- 2026-01-01 至 2026-11-08 的 6,964 场仅为候选测试清单。逐场输入尚未补齐，未计算正式测试成绩。
- 输入取窗口开始后前 29 根已收盘的分钟线，以及 `[T+24 分钟, T+29 分钟)` 内最后一笔有效 Polymarket 成交。时间核查基于历史事件时间。标签使用 Polymarket 最终结算。

## 模型结果

| 方法 | 8–11 月滚动 Brier | 12 月 Brier | 12 月准确率 |
|---|---:|---:|---:|
| Polymarket 成交概率 | **0.17230** | 0.17086 | 74.8% |
| L2 逻辑回归 | 0.17345 | **0.17016** | 75.8% |
| 市场概率修正 | 0.17232 | 0.17066 | 74.9% |
| 浅层 XGBoost | 0.17522 | 0.17214 | 74.2% |
| L2 + 成交量失衡 | 0.17358 | 0.17040 | 75.3% |

Brier 是概率误差的平方均值，越低越好。2025 年复核中，L2 的 Brier 略低于市场，差值按天重抽样得到的 95% 区间跨零。加入五分钟成交量失衡后，两段验证均未降低原 L2 的 Brier。

## 在本机复现

使用 Python 3.12。在仓库根目录运行：

```bash
python -m pip install -r requirements.txt
python src/reproduce.py
```

脚本以随仓库提供的 Parquet 为输入，重新调参和训练三组实验，并把新结果写入 `runs/`。运行结束会逐组核对场数、选参和主要指标。已交付的结果可在 [results/](results/) 查看。

[采集](src/collect.py)和[清洗](src/prepare.py)脚本也已纳入。需要从公开接口重新采集时，运行 `python src/collect.py`，然后运行 `python src/prepare.py`。重新采集依赖外部接口，返回结果可能随源站变化。采集和重建文件写入 `runs/`。

## 原始成交与附录

约 371 MB 的完整数据包单独交付，其文件名和 SHA-256 见 [数据包清单](data/raw_archive.json)。仓库里的特征表、覆盖表和两份 Binance 分钟线已足够重跑上述模型。原始来源见 [Polymarket 数据接口](https://data-api.polymarket.com/v2/openapi.json) 和 [Binance 公共数据档案](https://github.com/binance/binance-public-data)。

- [数据覆盖报告](reports/appendix/data_coverage.html)
- [L2 调参与公式](reports/appendix/l2_tuning.pdf)
- [市场修正与 XGBoost 调参](reports/appendix/offset_and_xgboost.pdf)
