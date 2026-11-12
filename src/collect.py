from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from hashlib import sha256
from http.client import HTTPSConnection
from pathlib import Path
from threading import Lock, local
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import argparse
import gzip
import json
import ssl
import sys
import time

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / 'runs/raw'
MANIFEST = REPO / 'data/selection_2025.json.gz'

HOST = 'data-api.polymarket.com'
TRADE_PATH = '/v2/trades'
MONTHS = [f'2025-{m:02}' for m in range(5, 13)]
FIELDS = ('timestamp', 'price', 'size', 'outcome', 'side', 'token_id', 'transaction_hash')
WORKERS = 28
MAX_RATE = 24
TRADE_PARAMS = {'limit': 1000, 'taker_only': 'true'}
CONTEXT = ssl.create_default_context()
THREAD = local()
GATE = Lock()
next_request = 0.0
retry_counts = Counter()
sys.stdout.reconfigure(encoding='utf-8')


def write_json(path, value):
    partial = path.with_suffix(path.suffix + '.tmp')
    partial.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    partial.replace(path)


def seconds(value):
    return int(datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp())


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def get_selection():
    with gzip.open(MANIFEST, 'rt', encoding='utf-8') as archive:
        rows = json.load(archive)
    if dict(Counter(row['split'] for row in rows)) != {'train': 4438, 'validation': 738}:
        raise ValueError('The frozen split manifest changed')
    return rows


def archive_fetch(url):
    for attempt in range(4):
        try:
            request = Request(url, headers={'User-Agent': 'Mozilla/5.0', 'Accept-Encoding': 'gzip'})
            with urlopen(request, timeout=18) as response:
                data = response.read()
                return gzip.decompress(data) if response.headers.get('Content-Encoding') == 'gzip' else data
        except Exception as error:
            if attempt == 3 or (isinstance(error, HTTPError) and error.code not in (429, 500, 502, 503, 504)):
                raise
            retry_after = error.headers.get('Retry-After', '') if isinstance(error, HTTPError) else ''
            time.sleep(min(60, int(retry_after)) if retry_after.isdigit() else 2 ** attempt)


def fetch_archive(venue, month):
    name = f'{venue}_BTCUSDT-1m-{month}.zip'
    path = ROOT / 'binance' / name
    receipt = path.with_suffix('.json')
    if path.exists() and receipt.exists():
        info = json.loads(receipt.read_text(encoding='utf-8'))
        if sha256(path.read_bytes()).hexdigest() == info['sha256']:
            return info
    prefix = 'spot' if venue == 'spot' else 'futures/um'
    url = f'https://data.binance.vision/data/{prefix}/monthly/klines/BTCUSDT/1m/{name.split("_", 1)[1]}'
    digest = archive_fetch(url + '.CHECKSUM').decode('ascii').split()[0].lower()
    body = archive_fetch(url)
    if sha256(body).hexdigest() != digest:
        raise ValueError(f'Checksum mismatch: {url}')
    path.write_bytes(body)
    info = {'url': url, 'file': name, 'sha256': digest, 'bytes': len(body), 'fetched_utc': utc_now()}
    write_json(receipt, info)
    return info


def pace():
    global next_request
    with GATE:
        start = max(time.monotonic(), next_request)
        next_request = start + 1 / MAX_RATE
        # A queued request checks the shared rate-limit pause before dispatch.
        time.sleep(max(0, start - time.monotonic()))


def get_page(params):
    global next_request
    target = TRADE_PATH + '?' + urlencode(params)
    for attempt in range(5):
        try:
            pace()
            if getattr(THREAD, 'conn', None) is None:
                THREAD.conn = HTTPSConnection(HOST, timeout=18, context=CONTEXT)
            THREAD.conn.request('GET', target, headers={'Accept': 'application/json', 'Accept-Encoding': 'gzip', 'User-Agent': 'Mozilla/5.0'})
            response = THREAD.conn.getresponse()
            body = response.read()
            if response.status == 200:
                if response.getheader('Content-Encoding') == 'gzip':
                    body = gzip.decompress(body)
                return json.loads(body), sha256(body).hexdigest()
            if response.status not in (429, 500, 502, 503, 504):
                raise ValueError(f'HTTP {response.status} for condition {params["condition"]}')
            retry = response.getheader('Retry-After', '')
            delay = min(120, int(retry)) if retry.isdigit() else min(30, 2 ** attempt)
            with GATE:
                retry_counts[f'http_{response.status}'] += 1
                if response.status == 429 or (response.status == 503 and retry.isdigit()):
                    next_request = max(next_request, time.monotonic() + delay)
            if attempt == 4:
                raise ValueError(f'HTTP {response.status} after retries')
            time.sleep(delay)
        except (OSError, TimeoutError, ConnectionError, EOFError) as error:
            if getattr(THREAD, 'conn', None) is not None:
                THREAD.conn.close()
                THREAD.conn = None
            with GATE:
                retry_counts[type(error).__name__] += 1
            if attempt == 4:
                raise
            time.sleep(min(10, 1 + 2 * attempt))


def cache_path(row):
    return ROOT / 'trade_windows' / row['start_bj'][:7] / f'{row["event_id"]}.json.gz'


def download_window(row):
    start, end = seconds(row['start_utc']), seconds(row['end_utc'])
    path = cache_path(row)
    if path.exists():
        receipt = json.loads(gzip.decompress(path.read_bytes()))['receipt']
        if receipt['condition_id'] == row['condition_id'] and [receipt['start_s'], receipt['end_s']] == [start, end]:
            return receipt
        raise ValueError(f'Cache identity mismatch: {path.name}')
    params = {'condition': row['condition_id'], **TRADE_PARAMS}
    rows, page_hashes, cursors = [], [], set()
    prev = float('inf')
    count, after, before = 0, 0, 0
    for page in range(1, 121):
        payload, digest = get_page(params)
        page_hashes.append(digest)
        trades = payload['data']
        for trade in trades:
            stamp = trade['timestamp']
            if not isinstance(stamp, int) or stamp > prev or trade['condition_id'].lower() != row['condition_id'].lower():
                raise ValueError('Invalid timestamp order or market identity')
            prev = stamp
            count += 1
            if stamp >= end:
                after += 1
            elif stamp < start:
                before += 1
            else:
                rows.append([trade.get(key) for key in FIELDS])
        cursor = payload['pagination'].get('next_cursor')
        if not cursor or prev < start:
            stop = 'cursor_exhausted' if not cursor else 'crossed_window_start'
            break
        if cursor in cursors or not trades:
            raise ValueError('Pagination did not advance')
        cursors.add(cursor)
        params['cursor'] = cursor
    else:
        raise ValueError('Exceeded 120 pages before reaching the prediction window start')
    assert count == before + after + len(rows)
    receipt = {'event_id': row['event_id'], 'condition_id': row['condition_id'], 'start_s': start, 'end_s': end,
               'pages': page, 'stop': stop, 'seen_trades': count, 'after_window': after, 'before_window': before,
               'window_trades': len(rows), 'fetched_utc': utc_now(), 'page_sha256': page_hashes}
    partial = path.with_suffix('.gz.tmp')
    body = json.dumps({'receipt': receipt, 'fields': FIELDS, 'trades': rows}, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf-8')
    partial.write_bytes(gzip.compress(body, compresslevel=3))
    partial.replace(path)
    return receipt


def main():
    global ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument('--archives-only', action='store_true')
    parser.add_argument('--raw-dir', type=Path, default=ROOT)
    args = parser.parse_args()
    ROOT = args.raw_dir.resolve()
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / 'binance').mkdir(exist_ok=True)
    rows = get_selection()
    for month in MONTHS:
        (ROOT / 'trade_windows' / month).mkdir(parents=True, exist_ok=True)
    write_json(ROOT / 'selection.json', rows)
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=6) as archive_pool, ThreadPoolExecutor(max_workers=WORKERS) as market_pool:
        archive_jobs = [archive_pool.submit(fetch_archive, venue, month) for venue in ('spot', 'perp') for month in MONTHS]
        jobs = {} if args.archives_only else {market_pool.submit(download_window, row): row for row in rows}
        print(f'2025 scope: {len(rows)} windows, {len(archive_jobs)} Binance monthly archives. Cached windows resume automatically.', flush=True)
        issues, receipts, by_split = {}, [], Counter()
        for done, future in enumerate(as_completed(jobs), 1):
            row = jobs[future]
            try:
                receipt = future.result()
                receipts.append(receipt)
                by_split[row['split']] += 1
            except Exception as error:
                issues[row['event_id']] = str(error)
            if done % 200 == 0 or done == len(jobs):
                progress = {'processed': done, 'total': len(jobs), 'completed': len(receipts), 'errors': len(issues),
                            'completed_by_split': dict(by_split), 'elapsed_s': round(time.monotonic() - start), 'updated_utc': utc_now()}
                write_json(ROOT / 'progress.json', progress)
                print(json.dumps(progress), flush=True)
        archives = [future.result() for future in archive_jobs]
    write_json(ROOT / 'collection.json', {'archives': archives, 'issues': issues, 'completed_by_split': dict(by_split),
                                          'trade_requests_pages': sum(row['pages'] for row in receipts),
                                          'retries': dict(retry_counts), 'completed_utc': utc_now(), 'windows_requested': len(jobs)})
    print(f'Collection completed: {len(receipts)} successful windows, {len(issues)} failed, {len(archives)} Binance archives.', flush=True)


if __name__ == '__main__':
    main()
