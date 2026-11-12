from collections import Counter, defaultdict
from csv import reader
from datetime import datetime, timezone
from hashlib import sha256
from html import escape
from io import TextIOWrapper
from pathlib import Path
from zipfile import ZipFile, ZIP_STORED
import argparse
import gzip
import json
import math
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / 'runs/raw'
EXPORT = REPO / 'runs/rebuilt_data'
BINANCE_CUTOFF = int(datetime.fromisoformat('2025-12-31T16:00:00+00:00').timestamp())
TRAIN_END = '2025-11-30T16:00:00Z'
MODEL_FIELDS = ('spot_return_29m_bp', 'spot_return_5m_bp', 'spot_realized_vol_29m_bp',
                'perp_basis_bp', 'perp_basis_change_bp', 'poly_logit_last')
PROBABILITY_FLOOR = 0.005
TRADE_FIELDS = ('timestamp', 'price', 'size', 'outcome', 'side', 'token_id', 'transaction_hash')
sys.stdout.reconfigure(encoding='utf-8')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def epoch(value):
    return int(datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp())


def iso(seconds):
    return datetime.fromtimestamp(seconds, timezone.utc).isoformat().replace('+00:00', 'Z')


def normalize_binance(venue, archives, lower):
    # Official spot archive uses microseconds in 2025, USD-M uses milliseconds.
    scale = 1_000_000 if venue == 'spot' else 1_000
    multiplier = 1_000_000 // scale
    names = ('open_time_utc', 'open', 'high', 'low', 'close', 'volume_btc', 'close_time_utc',
             'quote_volume', 'trade_count', 'taker_buy_volume_btc', 'taker_buy_quote_volume')
    time_type = pa.timestamp('us', tz='UTC')
    schema = pa.schema(list(zip(names, (time_type, *([pa.float64()] * 5), time_type, pa.float64(), pa.int64(), pa.float64(), pa.float64()))))
    bars = {}
    file = EXPORT / f'binance_{venue}_BTCUSDT_1m.parquet'
    month_counts = {}
    with pq.ParquetWriter(file, schema, compression='zstd') as writer:
        for item in sorted(archives, key=lambda item: item['file']):
            if not item['file'].startswith(venue + '_'):
                continue
            archive = ROOT / 'binance' / item['file']
            if sha256(archive.read_bytes()).hexdigest() != item['sha256']:
                raise ValueError(f'Checksum changed: {archive.name}')
            batch = []
            with ZipFile(archive) as zf, zf.open(zf.namelist()[0]) as file_data:
                for fields in reader(TextIOWrapper(file_data, encoding='utf-8-sig')):
                    if not fields or not fields[0].isdigit():
                        continue
                    timestamp = int(fields[0])
                    start = timestamp // scale
                    if start < lower or start >= BINANCE_CUTOFF:
                        continue
                    close_time = int(fields[6])
                    if timestamp % (60 * scale) or close_time != timestamp + 60 * scale - 1 or start in bars:
                        raise ValueError(f'Unexpected or duplicate Binance minute in {archive.name}: {timestamp}')
                    values = [timestamp * multiplier, *(float(x) for x in fields[1:6]), close_time * multiplier,
                              float(fields[7]), int(fields[8]), float(fields[9]), float(fields[10])]
                    if min(values[1:5]) <= 0 or not all(math.isfinite(x) for x in values):
                        raise ValueError(f'Non-finite or invalid Binance prices: {archive.name}, {timestamp}')
                    batch.append(dict(zip(names, values)))
                    bars[start] = (values[1], values[4])
            month_counts[archive.name[-11:-4]] = len(batch)
            if batch:
                writer.write_table(pa.Table.from_pylist(batch, schema=schema))
    return bars, month_counts


def model_features(row, start, first_trade, spot, perp):
    stamps = [start + minute * 60 for minute in range(29)]
    price = float(first_trade[1]) if first_trade[3] == 'Up' else 1 - float(first_trade[1])
    clipped = min(1 - PROBABILITY_FLOOR, max(PROBABILITY_FLOOR, price))
    spot_open, perp_open = spot[start][0], perp[start][0]
    spot_close, perp_close = spot[stamps[-1]][1], perp[stamps[-1]][1]
    returns = np.diff(np.log([spot_open, *(spot[t][1] for t in stamps)]))
    basis = 10000 * math.log(perp_close / spot_close)
    return {
        'event_id': row['event_id'], 'condition_id': row['condition_id'], 'slug': row['slug'],
        'split': row['split'], 'start_utc': row['start_utc'], 'start_bj': row['start_bj'], 'date': row['start_bj'][:10],
        'feature_cut_s': start + 29 * 60, 'decision_s': start + 30 * 60,
        'poly_trade_timestamp_s': first_trade[0], 'poly_probability': price,
        'target_up': int(row['winner'] == 'Up'),
        'spot_settlement_audit_up': int(spot[start + 59 * 60][1] >= spot_open),
        'spot_return_29m_bp': 10000 * math.log(spot_close / spot_open),
        'spot_return_5m_bp': 10000 * math.log(spot_close / spot[stamps[23]][1]),
        'spot_realized_vol_29m_bp': 10000 * float(np.linalg.norm(returns)),
        'perp_basis_bp': basis,
        'perp_basis_change_bp': basis - 10000 * math.log(perp_open / spot_open),
        'poly_logit_last': math.log(clipped / (1 - clipped)),
    }


def valid_trade(record, row):
    stamp, price, size, outcome, side, token, _ = record
    return (outcome in ('Up', 'Down') and token == row['token_' + outcome.lower()]
            and isinstance(stamp, int) and price is not None and 0 <= float(price) <= 1)


def read_window(row, collection_issues):
    path = ROOT / 'trade_windows' / row['start_bj'][:7] / f'{row["event_id"]}.json.gz'
    if not path.is_file():
        return None, [], collection_issues.get(row['event_id'], 'No complete cached response')
    record = json.loads(gzip.decompress(path.read_bytes()))
    receipt, trades = record['receipt'], record['trades']
    start, end = epoch(row['start_utc']), epoch(row['end_utc'])
    if (tuple(record['fields']) != TRADE_FIELDS or receipt['condition_id'] != row['condition_id']
            or receipt['window_trades'] != len(trades) or (receipt['start_s'], receipt['end_s']) != (start, end)
            or receipt['seen_trades'] != len(trades) + receipt['before_window'] + receipt['after_window']
            or receipt['stop'] not in ('cursor_exhausted', 'crossed_window_start')
            or receipt['pages'] != len(receipt['page_sha256'])):
        raise ValueError(f'Unverified trade receipt: {path.name}')
    if any(not start <= trade[0] < end for trade in trades) or any(left[0] < right[0] for left, right in zip(trades, trades[1:])):
        raise ValueError(f'Bad trade timestamps: {path.name}')
    return receipt, trades, ''


def export_trades(selected, issues, spot, perp):
    schema = pa.schema([('event_id', pa.string()), ('timestamp_utc', pa.timestamp('s', tz='UTC')),
                        ('order_in_window_desc', pa.int32()), ('price', pa.float64()), ('shares', pa.float64()),
                        ('outcome', pa.string()), ('side', pa.string()), ('token_id', pa.string()), ('transaction_hash', pa.string())])
    coverage, features = [], []
    (EXPORT / 'polymarket_trades').mkdir(exist_ok=True)
    month_groups = defaultdict(list)
    for row in selected:
        month_groups[row['start_bj'][:7]].append(row)
    for month, group in sorted(month_groups.items()):
        file = EXPORT / 'polymarket_trades' / f'{month}.parquet'
        batch = []
        with pq.ParquetWriter(file, schema, compression='zstd') as writer:
            for row in group:
                start, end = epoch(row['start_utc']), epoch(row['end_utc'])
                if start % 3600 or end - start != 3600 or not row['start_bj'].startswith('2025-') or end > BINANCE_CUTOFF:
                    raise ValueError(f'Bad selected hour: {row["event_id"]}')
                receipt, trades, error = read_window(row, issues)
                row_valid_trades = [t for t in trades if valid_trade(t, row)]
                feature_trades = [t for t in row_valid_trades if start + 24 * 60 <= t[0] < start + 29 * 60]
                minutes = range(start, end, 60)
                spot_count = sum(minute in spot for minute in minutes)
                perp_count = sum(minute in perp for minute in minutes)
                first_29 = range(start, start + 29 * 60, 60)
                model_ok = all(t in spot and t in perp for t in first_29) and start + 59 * 60 in spot
                reason = ('polymarket_download_failed' if error else 'binance_model_minutes_missing' if not model_ok
                          else 'no_polymarket_trade_minute24_29' if not feature_trades else '')
                if not reason:
                    features.append(model_features(row, start, feature_trades[0], spot, perp))
                audit_winner = ('Up' if spot[start + 59 * 60][1] >= spot[start][0] else 'Down') if start in spot and start + 59 * 60 in spot else None
                market_meta = {key: value for key, value in row.items() if key != 'excluded_reasons'}
                coverage.append({**market_meta, 'download_status': 'failed' if error else 'empty' if not trades else 'downloaded',
                                 'download_error': error, 'window_trade_count': len(trades) if receipt else None,
                                 'first_trade_utc': iso(trades[-1][0]) if trades else None,
                                 'last_trade_utc': iso(trades[0][0]) if trades else None,
                                 'api_pages': receipt['pages'] if receipt else None, 'pagination_end': receipt['stop'] if receipt else None,
                                 'valid_trade_count': len(row_valid_trades) if receipt else None,
                                 'feature_trade_count': len(feature_trades) if receipt else None,
                                 'spot_minutes': spot_count, 'perp_minutes': perp_count, 'binance_spot_winner': audit_winner,
                                 'model_ready': not reason, 'model_exclusion_reason': reason})
                for ordinal, (stamp, price, size, outcome, side, token, tx) in enumerate(trades):
                    batch.append({'event_id': row['event_id'], 'timestamp_utc': stamp, 'order_in_window_desc': ordinal,
                                  'price': float(price) if price is not None else None, 'shares': float(size) if size is not None else None,
                                  'outcome': outcome, 'side': side, 'token_id': token, 'transaction_hash': tx})
                if len(batch) >= 75_000:
                    writer.write_table(pa.Table.from_pylist(batch, schema=schema))
                    batch.clear()
            if batch:
                writer.write_table(pa.Table.from_pylist(batch, schema=schema))
        print(month, f'{len(group)} markets normalized', flush=True)
    return coverage, features


def check_pilot(features):
    # Verify that overlapping inputs retain the original pilot's feature convention.
    pilot_path = REPO / 'data/reference/pilot_samples.json.gz'
    with gzip.open(pilot_path, 'rt', encoding='utf-8') as file:
        pilot = {row['event_id']: row for row in json.load(file)}
    common = [row for row in features if row['event_id'] in pilot]
    changed = []
    for row in common:
        previous = pilot[row['event_id']]
        if not all(math.isclose(row[key], previous[key], rel_tol=1e-12, abs_tol=1e-11) for key in MODEL_FIELDS + ('poly_probability', 'target_up', 'poly_trade_timestamp_s')):
            changed.append({'event_id': row['event_id'], 'start_bj': row['start_bj']})
    if not common:
        raise ValueError('No overlap with pilot features for validation')
    return {'common': len(common), 'identical': len(common) - len(changed), 'changed': changed}


def summarize(coverage, features, candle_counts, collection, pilot_check):
    groups = defaultdict(list)
    for row in coverage:
        groups[row['split']].append(row)
    splits = {}
    for key, items in groups.items():
        matches = [r for r in items if r['binance_spot_winner'] is not None]
        splits[key] = {'selected': len(items), 'first_bj': items[0]['start_bj'], 'last_bj': items[-1]['start_bj'],
                       'polymarket_complete': sum(r['download_status'] != 'failed' for r in items),
                       'polymarket_nonempty': sum(r['download_status'] == 'downloaded' for r in items),
                       'polymarket_rows': sum(r['window_trade_count'] or 0 for r in items),
                       'both_full_binance_hours': sum(r['spot_minutes'] == r['perp_minutes'] == 60 for r in items),
                       'model_ready': sum(r['model_ready'] for r in items),
                       'spot_label_checked': len(matches), 'spot_label_disagreements': sum(r['winner'] != r['binance_spot_winner'] for r in matches),
                       'exclusions': dict(Counter(r['model_exclusion_reason'] for r in items if not r['model_ready']))}
    months = defaultdict(Counter)
    for row in coverage:
        group = months[row['start_bj'][:7]]
        group['selected'] += 1
        group['polymarket_complete'] += row['download_status'] != 'failed'
        group['polymarket_rows'] += row['window_trade_count'] or 0
        group['both_full_binance_hours'] += row['spot_minutes'] == row['perp_minutes'] == 60
        group['model_ready'] += row['model_ready']
    return {'splits': splits, 'months': dict(months), 'features': len(features), 'binance_minutes': candle_counts,
            'binance_archives': collection['archives'], 'polymarket_pages': collection['trade_requests_pages'],
            'source_failures': collection['issues'], 'retries': collection['retries'], 'pilot_cross_check': pilot_check,
            'exported_at_utc': datetime.now(timezone.utc).isoformat(), 'arrow_version': pa.__version__}


def table(headers, rows):
    header = ''.join(f'<th>{escape(str(cell))}</th>' for cell in headers)
    body = ''.join('<tr>' + ''.join(f'<td>{escape(str(cell))}</td>' for cell in row) + '</tr>' for row in rows)
    return f'<div class="wrap"><table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table></div>'


def report(result):
    num = lambda value: f'{value:,}'
    splits = result['splits']
    cols = ('selected', 'polymarket_complete', 'polymarket_rows', 'both_full_binance_hours', 'model_ready')
    split_rows = [(label, *(num(splits[key][col]) for col in cols)) for key, label in [('train', '训练'), ('validation', '验证')]]
    months = [(month, *(num(row[col]) for col in cols)) for month, row in result['months'].items()]
    labels = {'polymarket_download_failed': '成交下载失败', 'binance_model_minutes_missing': '缺少模型所需的 Binance 分钟线',
              'no_polymarket_trade_minute24_29': '第 24—29 分钟没有有效 Polymarket 成交'}
    exclusions = [(split, labels.get(reason, reason), n) for split, row in splits.items() for reason, n in row['exclusions'].items()]
    labels_checked = sum(row['spot_label_checked'] for row in splits.values())
    disagreements = sum(row['spot_label_disagreements'] for row in splits.values())
    empty_windows = sum(row['polymarket_complete'] - row['polymarket_nonempty'] for row in splits.values())
    css = '''
:root{color-scheme:only light;background:#fff;color:#171717}html,body{background:#fff!important;color:#171717!important}
body{margin:35px auto;padding:0 24px 48px;max-width:1000px;font:17px/1.75 "Microsoft YaHei","PingFang SC",sans-serif}
h1{font-size:28px;line-height:1.4}h2{font-size:21px;margin-top:30px}.wrap{overflow-x:auto}
table{border-collapse:collapse;width:100%;background:#fff;font-size:15px}th,td{text-align:left;border:1px solid #d7dce2;padding:8px 10px;color:#171717}th{background:#f0f3f6}a{color:#0758a3}
'''
    columns = ['范围', '候选小时窗口', '成交分页已核对', '窗口内成交条数', '两边分钟线均满 60 根', '可进入首轮模型']
    html = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="color-scheme" content="light"><title>Polymarket 2025 小时盘数据补齐</title><style>{css}</style></head><body><main>
<h1>Polymarket 2025 小时盘数据补齐</h1>
<p>沿用已定的时间切分。训练从北京时间 {splits['train']['first_bj'][:10]} 至 2025-11-30，验证为 2025 年 12 月。Parquet 中的时间戳统一为 UTC，切分按北京时间。市场目录在同一个小时有重复合约时，只取候选中事件编号较早的一个。</p>
{table(columns, split_rows)}
<p>Polymarket 每个合约按成交接口逐页下载，保留预测窗口内的记录，窗口起点包含、终点排除。分页达到窗口开始前或接口末页视为覆盖已核对，其中窗口内无成交的有 {empty_windows} 场。成交保留时间、价格、份额、Up／Down、方向、代币编号和交易哈希。用 taker 侧计数，交易哈希可能对应多笔记录。</p>
<p>Binance BTCUSDT 现货和 USDT 永续共核验 {len(result['binance_archives'])} 份月度档案。分钟线和 Polymarket 窗口按 UTC 对齐。Binance 完整小时的涨跌核对了 {num(labels_checked)} 场，差异 {num(disagreements)} 场。标签以 Polymarket 已结算结果为准。</p>
<h2>为什么有的窗口不能进模型</h2>
{table(['集合','原因','场数'], exclusions) if exclusions else '<p>候选窗口均通过首轮模型的数据检查。</p>'}
<p>首轮模型仍在每小时开始后第 30 分钟作出预测。Polymarket 取第 24 分钟（含）至第 29 分钟（不含）最后一笔有效成交，Binance 取截至第 29 分钟已收盘的分钟线。缺失值保留为空。与上一轮样本重合的 {result['pilot_cross_check']['common']} 场中，{result['pilot_cross_check']['identical']} 场的输入及标签一致。Polymarket 成交使用事件时间，历史收到时间和盘口未纳入。</p>
<h2>按月核对</h2>
{table(columns, months)}
<p>数据包含按月 Polymarket 成交 Parquet、Binance 两份分钟线、逐场覆盖清单和已经对齐的模型输入。完整分钟线和成交会涵盖预测后时段，训练时应从对齐后的 <code>model_features.parquet</code> 读取特征。<a href="https://data-api.polymarket.com/v2/openapi.json">Polymarket 官方接口</a> · <a href="https://github.com/binance/binance-public-data">Binance 官方档案</a></p>
</main></body></html>'''
    (EXPORT / '2025小时盘数据补齐报告.html').write_text(html, encoding='utf-8')


def main():
    global pa, pq, ROOT, EXPORT
    parser = argparse.ArgumentParser()
    parser.add_argument('--arrow-path')
    parser.add_argument('--raw-dir', type=Path, default=ROOT)
    parser.add_argument('--output-dir', type=Path, default=EXPORT)
    args = parser.parse_args()
    ROOT = args.raw_dir.resolve()
    EXPORT = args.output_dir.resolve()
    if EXPORT == (REPO / 'data').resolve():
        parser.error('Choose a separate output directory for rebuilt data')
    if args.arrow_path:
        sys.path.append(args.arrow_path)
    import pyarrow as pa
    import pyarrow.parquet as pq
    EXPORT.mkdir(parents=True, exist_ok=True)
    selected = read_json(ROOT / 'selection.json')
    collection = read_json(ROOT / 'collection.json')
    if len(selected) != collection['windows_requested'] or len(collection['archives']) != 16:
        raise ValueError('Collection and market manifest do not match')
    if len({row['start_utc'] for row in selected}) != len(selected):
        raise ValueError('Duplicate hour in the market manifest')
    lower = min(epoch(row['start_utc']) for row in selected)
    spot, spot_months = normalize_binance('spot', collection['archives'], lower)
    perp, perp_months = normalize_binance('perp', collection['archives'], lower)
    coverage, features = export_trades(selected, collection['issues'], spot, perp)
    for row in features:
        assert row['poly_trade_timestamp_s'] < row['feature_cut_s'] < row['decision_s']
        assert row['start_bj'].startswith('2025-')
        assert (row['start_utc'] < TRAIN_END) == (row['split'] == 'train')
    check = check_pilot(features)
    summary = summarize(coverage, features, {'spot': spot_months, 'perp': perp_months}, collection, check)
    pq.write_table(pa.Table.from_pylist(coverage), EXPORT / 'market_coverage.parquet', compression='zstd')
    pq.write_table(pa.Table.from_pylist(features), EXPORT / 'model_features.parquet', compression='zstd')
    write_json(EXPORT / 'coverage_summary.json', summary)
    report(summary)
    readme = '''2025 Polymarket BTC 小时盘数据\n\n按北京时间划分训练和验证，日期截至 2025 年 12 月 31 日。Parquet 时间字段为 UTC。\n\n- polymarket_trades/：按窗口开始时间的北京时间月份分文件，event_id 可与 market_coverage 关联。order_in_window_desc 从 0 开始，保留接口在同一秒中的返回顺序。shares 是预测份额，价格为该份额的成交价。\n- binance_spot_BTCUSDT_1m.parquet / binance_perp_BTCUSDT_1m.parquet：每分钟 OHLC、成交量、成交笔数，截到北京时间 2026-01-01 之前。\n- market_coverage.parquet：一场一行，含切分、窗口起止、结算结果、数据缺失原因及对齐统计。\n- model_features.parquet：一场一行，复现第 30 分钟预测口径。用 split 选训练和验证，target_up 为标签。spot_settlement_audit_up 只用于校核，建模输入请限于下列六列。\n- coverage_summary.json：按月和集合统计、校验摘要及档案来源。\n\n模型输入列：\n'''
    (EXPORT / '数据说明.txt').write_text(readme + '\n'.join(MODEL_FIELDS) + '\n', encoding='utf-8')
    totals = sum(pq.ParquetFile(path).metadata.num_rows for path in (EXPORT / 'polymarket_trades').glob('*.parquet'))
    assert totals == sum(item['polymarket_rows'] for item in summary['splits'].values())
    assert pq.ParquetFile(EXPORT / 'model_features.parquet').metadata.num_rows == len(features)
    assert pq.ParquetFile(EXPORT / 'market_coverage.parquet').metadata.num_rows == len(selected)
    destination = EXPORT.parent / (EXPORT.name + '.zip')
    with ZipFile(destination, 'w', ZIP_STORED, allowZip64=True) as zip_file:
        for file in sorted(EXPORT.rglob('*')):
            if file.is_file():
                zip_file.write(file, file.relative_to(EXPORT).as_posix())
    with ZipFile(destination) as zip_file:
        if zip_file.testzip() is not None:
            raise ValueError('Package integrity check failed')
    print(json.dumps({'splits': summary['splits'], 'pilot': check, 'zip_bytes': destination.stat().st_size}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
