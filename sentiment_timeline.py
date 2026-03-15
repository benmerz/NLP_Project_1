#!/usr/bin/env python3
"""
Sentiment Analysis Over Time
Loads chat logs (HTML, CSV, JSON, JSONL), scores each message with a
transformer, and plots average sentiment per time period.
"""

import re
import json
import argparse
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from bs4 import BeautifulSoup
from transformers import pipeline
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Timestamp parsing
# ---------------------------------------------------------------------------

_TS_RE = re.compile(
    r'([A-Z][a-z]{2,3}) (\d{1,2}), (\d{4}), (\d{1,2}):(\d{2}):(\d{2})[\s\u202f]+([AP]M)'
)


def parse_timestamp(ts: str):
    """Parse a Google Takeout-style timestamp string; returns datetime or None."""
    if not ts or ts == 'N/A':
        return None
    m = _TS_RE.search(ts)
    if not m:
        return None
    month, day, year, hour, minute, second, ampm = m.groups()
    try:
        return datetime.strptime(f"{month} {day} {year} {hour}:{minute}:{second} {ampm}",
                                 "%b %d %Y %I:%M:%S %p")
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# File loaders  (each returns a list of {"speaker", "text", "timestamp"} dicts)
# ---------------------------------------------------------------------------

def load_html(path: str):
    """Parse Google Takeout HTML (Gemini Apps export)."""
    _TS_HTML_RE = re.compile(
        r'([A-Z][a-z]{2,3} \d{1,2}, \d{4}, \d{1,2}:\d{2}:\d{2}[\s\u202f]+[AP]M [A-Z]{3})'
    )
    with open(path, encoding='utf-8') as f:
        soup = BeautifulSoup(f.read(), 'html.parser')

    messages = []
    for cell in soup.find_all('div', class_='outer-cell'):
        content = cell.find('div', class_=lambda x: x and 'content-cell' in x
                            and 'mdl-cell--6-col' in x and 'mdl-typography--body-1' in x)
        if not content:
            continue
        children = list(content.children)
        first = str(children[0]) if children else ''
        if 'Prompted' not in first:
            continue

        prompted = first.split('Prompted', 1)[1].strip()
        parts = prompted.split('\n\n', 1)
        user_text = parts[1].strip() if len(parts) == 2 else parts[0].strip()

        timestamp, ai_parts, found = 'N/A', [], False
        for child in children:
            s = str(child).strip()
            if not found:
                m = _TS_HTML_RE.search(s)
                if m:
                    timestamp = m.group(1)
                    found = True
                continue
            if hasattr(child, 'get_text'):
                t = child.get_text(separator=' ', strip=True)
                if t:
                    ai_parts.append(t)

        ai_text = ' '.join(ai_parts)
        if user_text:
            messages.append({'speaker': 'User', 'text': user_text, 'timestamp': timestamp})
        if ai_text:
            messages.append({'speaker': 'AI',   'text': ai_text,   'timestamp': timestamp})

    return messages


def load_csv(path: str):
    df = pd.read_csv(path)
    if 'speaker' not in df.columns or 'text' not in df.columns:
        raise ValueError("CSV must have 'speaker' and 'text' columns")
    return [
        {'speaker': str(r['speaker']), 'text': str(r['text']),
         'timestamp': str(r.get('timestamp', 'N/A'))}
        for r in df.to_dict('records')
    ]


def load_json(path: str):
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("JSON must be a list of objects")
    messages = []
    for item in data:
        ts = str(item.get('timestamp', 'N/A'))
        if 'speaker' in item and 'text' in item:
            messages.append({'speaker': str(item['speaker']), 'text': str(item['text']), 'timestamp': ts})
        elif 'instruction' in item or 'response' in item:
            if item.get('instruction'):
                messages.append({'speaker': 'User', 'text': item['instruction'].strip(), 'timestamp': ts})
            if item.get('response'):
                messages.append({'speaker': 'AI',   'text': item['response'].strip(),    'timestamp': ts})
    return messages


def load_jsonl(path: str, limit=None):
    messages, count = [], 0
    with open(path, encoding='utf-8') as f:
        for line in f:
            if limit and count >= limit:
                break
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if item.get('instruction'):
                messages.append({'speaker': 'User', 'text': item['instruction'].strip(), 'timestamp': 'N/A'})
            if item.get('response'):
                messages.append({'speaker': 'AI',   'text': item['response'].strip(),    'timestamp': 'N/A'})
            count += 1
    return messages


def load_messages(path: str, limit=None):
    ext = Path(path).suffix.lower()
    loaders = {'.html': load_html, '.csv': load_csv, '.json': load_json}
    if ext in loaders:
        msgs = loaders[ext](path)
        return msgs[:limit * 2] if limit else msgs
    if ext == '.jsonl':
        return load_jsonl(path, limit=limit)
    raise ValueError(f"Unsupported file type: {ext}  (use .html, .csv, .json, .jsonl)")


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

_HTML_TAG = re.compile(r'<[^>]+>')
_URL      = re.compile(r'http\S+')
_SPACE    = re.compile(r'\s+')


def clean(text: str) -> str:
    text = _HTML_TAG.sub('', text)
    text = _URL.sub('', text)
    return _SPACE.sub(' ', text).strip()


def preprocess(messages: list, min_len=10) -> list:
    out = []
    for msg in messages:
        t = clean(msg['text'])
        if len(t) >= min_len:
            out.append({**msg, 'text': t})
    return out


# ---------------------------------------------------------------------------
# Sentiment analysis
# ---------------------------------------------------------------------------

def build_pipeline():
    import torch
    device = 0 if torch.cuda.is_available() else -1
    label  = 'GPU' if device == 0 else 'CPU'
    dtype  = torch.float16 if device == 0 else torch.float32
    print(f"Loading model (using {label})...")
    nlp = pipeline(
        'sentiment-analysis',
        model='lxyuan/distilbert-base-multilingual-cased-sentiments-student',
        truncation=True, max_length=512, device=device, torch_dtype=dtype
    )
    try:
        nlp.model = torch.compile(nlp.model)
    except Exception:
        pass
    nlp('warm up', batch_size=1)
    return nlp


def score_messages(messages: list, nlp, batch_size=64) -> pd.DataFrame:
    _map = {'positive': 1, 'negative': -1, 'neutral': 0}
    texts = [m['text'][:2000] for m in messages]
    raw = []
    for i in tqdm(range(0, len(texts), batch_size), desc='Scoring'):
        raw.extend(nlp(texts[i:i + batch_size]))

    rows = []
    for msg, r in zip(messages, raw):
        rows.append({
            'speaker':         msg['speaker'],
            'timestamp':       msg['timestamp'],
            'datetime_parsed': parse_timestamp(msg['timestamp']),
            'sentiment_score': r['score'] * _map.get(r['label'], 0),
            'sentiment_label': r['label'],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Time-series plot
# ---------------------------------------------------------------------------

COLORS = {'User': '#3b82f6', 'AI': '#10b981'}


def plot_over_time(df: pd.DataFrame, output_path='sentiment_over_time.png'):
    speakers = sorted(df['speaker'].unique())
    has_ts   = df['datetime_parsed'].notna().any()

    fig, ax = plt.subplots(figsize=(14, 6))

    if has_ts:
        df_ts = df[df['datetime_parsed'].notna()].copy()
        span  = (df_ts['datetime_parsed'].max() - df_ts['datetime_parsed'].min()).days
        freq  = 'W' if span <= 60 else ('M' if span <= 730 else 'Q')
        label = {'W': 'Week', 'M': 'Month', 'Q': 'Quarter'}[freq]

        df_ts['period'] = df_ts['datetime_parsed'].dt.to_period(freq)
        grouped = (df_ts.groupby(['period', 'speaker'])['sentiment_score']
                   .mean().reset_index())
        grouped['x'] = grouped['period'].dt.to_timestamp()

        for spk in speakers:
            sub = grouped[grouped['speaker'] == spk].sort_values('x')
            c   = COLORS.get(spk, '#6366f1')
            ax.plot(sub['x'], sub['sentiment_score'], marker='o', label=spk,
                    color=c, linewidth=2, markersize=5)
            ax.fill_between(sub['x'], sub['sentiment_score'], alpha=0.12, color=c)

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        fig.autofmt_xdate()
        ax.set_xlabel(label, fontsize=12, fontweight='bold')
        coverage = (f"{df_ts['datetime_parsed'].min().strftime('%b %Y')} – "
                    f"{df_ts['datetime_parsed'].max().strftime('%b %Y')}")
        title = f'Sentiment per {label}  ({coverage})'

        n_missing = df['datetime_parsed'].isna().sum()
        if n_missing:
            print(f"Note: {n_missing} messages had no timestamp and are excluded.")

    else:
        n = len(df)
        n_buckets = max(5, min(30, n // max(1, n // 20)))
        df = df.copy()
        df['bucket'] = pd.cut(df.index, bins=n_buckets, labels=False)
        grouped = (df.groupby(['bucket', 'speaker'])['sentiment_score']
                   .mean().reset_index())

        for spk in speakers:
            sub = grouped[grouped['speaker'] == spk].sort_values('bucket')
            c   = COLORS.get(spk, '#6366f1')
            ax.plot(sub['bucket'], sub['sentiment_score'], marker='o', label=spk,
                    color=c, linewidth=2, markersize=5)
            ax.fill_between(sub['bucket'], sub['sentiment_score'], alpha=0.12, color=c)

        ax.set_xlabel('Message Sequence (equal-size groups)', fontsize=12, fontweight='bold')
        title = f'Sentiment over Message Sequence  ({n} messages, {n_buckets} groups)'
        print("Note: No timestamps found — using message order as the time axis.")

    ax.set_ylabel('Average Sentiment Score', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_ylim(-1, 1)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(fontsize=11)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Sentiment over time from chat logs')
    parser.add_argument('input_file',               help='Input file (.html, .csv, .json, .jsonl)')
    parser.add_argument('-o', '--output',           default='sentiment_over_time.png',
                        help='Output plot path (default: sentiment_over_time.png)')
    parser.add_argument('--export',                 help='Export scored results to CSV')
    parser.add_argument('--min-length', type=int,   default=10,
                        help='Minimum message character length (default: 10)')
    parser.add_argument('--batch-size', type=int,   default=64,
                        help='Batch size for inference (default: 64)')
    parser.add_argument('--limit',      type=int,   default=None,
                        help='Max records to load (useful for quick tests)')
    args = parser.parse_args()

    if not Path(args.input_file).exists():
        print(f"Error: file not found: {args.input_file}")
        return

    print(f"Loading {args.input_file}...")
    messages = load_messages(args.input_file, limit=args.limit)
    print(f"  {len(messages)} messages loaded")

    messages = preprocess(messages, min_len=args.min_length)
    print(f"  {len(messages)} messages after preprocessing")

    if not messages:
        print("Error: no valid messages after preprocessing")
        return

    nlp = build_pipeline()
    df  = score_messages(messages, nlp, batch_size=args.batch_size)

    if args.export:
        df.to_csv(args.export, index=False)
        print(f"Results exported to: {args.export}")

    plot_over_time(df, output_path=args.output)


if __name__ == '__main__':
    main()
