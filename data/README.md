# 数据字典

划分按北京时间，K 线与带 `_utc` 的时间字段为 UTC，带 `_s` 的时间为 Unix 秒。成交量以 BTC 计，收益率、实现波动率和基差以 bp 计（1 bp = 0.01%）。

## 文件

| 文件 | 内容 |
|---|---|
| `model_features.parquet` | 每场一行的 2025 年建模特征，共 5,140 行。 |
| `market_coverage.parquet` | 5,176 场的窗口信息、成交覆盖、对齐检查和排除原因。 |
| `binance_spot_BTCUSDT_1m.parquet`、`binance_perp_BTCUSDT_1m.parquet` | BTCUSDT 现货和 USDT 永续的 1 分钟行情，含开收盘时间、OHLC、成交量、笔数和主动买入量。 |
| `coverage_summary.json`、`collection_receipts.json` | 覆盖统计、采集完成情况及 Binance 档案校验值。 |
| `selection_2025.json.gz` | 2025 年冻结的采集清单，供重新下载。 |
| `test_2026_manifest.json.gz` | 2026 年的 6,964 场候选窗口及标识，尚未对齐成测试特征。 |
| `raw_archive.json` | 单独交付的完整逐笔数据包的文件名、字节数和 SHA-256。 |
| `reference/` | 复算时用于核对的早期模型指标和重叠场次快照。 |

## 模型特征表

| 列 | 含义 |
|---|---|
| `event_id`、`condition_id`、`slug` | Polymarket 窗口标识及合约信息。 |
| `date`、`start_bj`、`start_utc`、`split` | 北京日期、窗口起点及训练/验证标记。 |
| `feature_cut_s`、`decision_s`、`poly_trade_timestamp_s` | 特征截止、决策、入选 Polymarket 成交的时间。 |
| `poly_probability` | 最近一笔有效成交换算到 Up 侧的价格。若该笔成交为 Down 份额，取 1 减成交价。直接使用该值作为市场基准。 |
| `target_up` | 最终结算标签，Up 为 1、Down 为 0。 |
| `spot_settlement_audit_up` | Binance 小时开收盘方向，仅用于交叉核对标签。 |
| `spot_return_29m_bp` | 从本小时开始到前 29 根分钟线收盘的现货对数收益。 |
| `spot_return_5m_bp` | 同一特征截止点前最后 5 根分钟线的现货对数收益。 |
| `spot_realized_vol_29m_bp` | 前 29 个分钟对数收益的平方和开方。 |
| `perp_basis_bp` | 截止时点的永续收盘价与现货收盘价之比取对数。 |
| `perp_basis_change_bp` | 上述基差减去小时开始时的基差。 |
| `poly_logit_last` | 将市场概率裁剪到 [0.005, 0.995] 后做 logit 变换。 |

六项基础输入是表中最后六行。成交量失衡扩展从两份 Binance 分钟线分别算出 `spot_trade_imbalance_5m` 和 `perp_trade_imbalance_5m`。区间为 [T+24 分钟, T+29 分钟)，公式为（主动买量 − 主动卖量）÷ 总量。一场永续窗口总量与成交笔数均为零，取中性值 0。

覆盖表中的 `download_status` 记录接口查询情况，`model_ready` 和 `model_exclusion_reason` 记录建模条件是否满足。查询完成且没有返回窗口内记录的场次标为 `empty`。输入和标签按历史事件时间核查，实际接收延迟未测得。
