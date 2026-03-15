#!/usr/bin/env python3
"""
LLM Chat Log Sentiment Analysis
Analyzes sentiment from chat logs (CSV, HTML, or JSON) and visualizes results.
"""

import re
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from bs4 import BeautifulSoup
import pandas as pd
import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from transformers import pipeline
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

_TIMESTAMP_PARSE_RE = re.compile(
    r'([A-Z][a-z]{2,3}) (\d{1,2}), (\d{4}), (\d{1,2}):(\d{2}):(\d{2})[\s\u202f]+([AP]M)'
)


def parse_timestamp(ts: str) -> Optional[datetime]:
    """
    Parse a Google Takeout-style timestamp string into a datetime object.

    Handles strings like "Dec 3, 2024, 9:06:45 AM EST" (timezone suffix is ignored).

    Args:
        ts: Raw timestamp string

    Returns:
        datetime object, or None if parsing fails
    """
    if not ts or ts == 'N/A':
        return None
    m = _TIMESTAMP_PARSE_RE.search(ts)
    if not m:
        return None
    month_str, day, year, hour, minute, second, ampm = m.groups()
    try:
        dt_str = f"{month_str} {day} {year} {hour}:{minute}:{second} {ampm}"
        return datetime.strptime(dt_str, "%b %d %Y %I:%M:%S %p")
    except ValueError:
        return None


class ChatLogParser:
    """Parse chat logs from various file formats"""

    _TIMESTAMP_RE = re.compile(
        r'([A-Z][a-z]{2,3} \d{1,2}, \d{4}, \d{1,2}:\d{2}:\d{2}[\s\u202f]+[AP]M [A-Z]{3})'
    )

    @staticmethod
    def parse_html(file_path: str) -> List[Dict[str, str]]:
        """
        Parse Google Takeout HTML format (Gemini Apps)

        Args:
            file_path: Path to HTML file

        Returns:
            List of dictionaries with 'speaker', 'text', and 'timestamp'
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')

        messages = []

        # Find all conversation cells
        conversation_cells = soup.find_all('div', class_='outer-cell')

        for cell in conversation_cells:
            # Find content div with flexible class matching
            content_div = cell.find('div', class_=lambda x: x and 'content-cell' in x and 'mdl-cell--6-col' in x and 'mdl-typography--body-1' in x)
            if not content_div:
                continue

            # Get children to parse structured content
            children = list(content_div.children)
            if not children:
                continue

            # First text node should contain "Prompted [AI_summary] [USER_question]"
            first_text = str(children[0]) if children else ""

            if 'Prompted' not in first_text:
                continue

            # Split by "Prompted" and handle the content after it
            parts = first_text.split('Prompted', 1)
            if len(parts) <= 1:
                continue

            prompted_content = parts[1].strip()

            # The user question is usually after a double newline
            # Format: AI_summary\n\nUSER_question
            content_parts = prompted_content.split('\n\n', 1)

            user_text = ""
            if len(content_parts) == 2:
                # There's a clear separation - second part is user question
                user_text = content_parts[1].strip()
            elif len(content_parts) == 1:
                # No separation - the whole thing might be the user question
                # This happens when there's no AI summary
                user_text = content_parts[0].strip()

            # Single pass: find timestamp then collect AI text
            timestamp = "N/A"
            ai_text_parts = []
            found_timestamp = False

            for child in children:
                child_str = str(child).strip()
                if not found_timestamp:
                    match = ChatLogParser._TIMESTAMP_RE.search(child_str)
                    if match:
                        timestamp = match.group(1)
                        found_timestamp = True
                    continue
                if hasattr(child, 'get_text'):
                    text = child.get_text(separator=' ', strip=True)
                    if text:
                        ai_text_parts.append(text)

            ai_text = ' '.join(ai_text_parts)

            # Add user message
            if user_text and len(user_text) > 10:
                messages.append({
                    'speaker': 'User',
                    'text': user_text,
                    'timestamp': timestamp
                })

            # Add AI message
            if ai_text and len(ai_text) > 10:
                messages.append({
                    'speaker': 'AI',
                    'text': ai_text,
                    'timestamp': timestamp
                })

        return messages

    @staticmethod
    def parse_csv(file_path: str) -> List[Dict[str, str]]:
        """
        Parse CSV format chat logs
        Expected columns: speaker, text, timestamp (timestamp is optional)

        Args:
            file_path: Path to CSV file

        Returns:
            List of dictionaries with 'speaker', 'text', and 'timestamp'
        """
        df = pd.read_csv(file_path)

        # Check for required columns
        if 'speaker' not in df.columns or 'text' not in df.columns:
            raise ValueError("CSV must contain 'speaker' and 'text' columns")

        return [
            {
                'speaker': str(row['speaker']),
                'text': str(row['text']),
                'timestamp': str(row.get('timestamp', 'N/A'))
            }
            for row in df.to_dict('records')
        ]

    @staticmethod
    def parse_jsonl(file_path: str, max_records: int = None) -> List[Dict[str, str]]:
        """
        Parse JSONL format (e.g. Databricks/Dolly dataset)
        Maps 'instruction' -> User, 'response' -> AI

        Args:
            file_path: Path to JSONL file
            max_records: Maximum number of JSONL records to read (optional)

        Returns:
            List of dictionaries with 'speaker', 'text', and 'timestamp'
        """
        messages = []
        records_read = 0
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if max_records is not None and records_read >= max_records:
                    break
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                instruction = item.get('instruction', '').strip()
                response = item.get('response', '').strip()
                if instruction:
                    messages.append({
                        'speaker': 'User',
                        'text': instruction,
                        'timestamp': 'N/A'
                    })
                if response:
                    messages.append({
                        'speaker': 'AI',
                        'text': response,
                        'timestamp': 'N/A'
                    })
                records_read += 1
        return messages

    @staticmethod
    def parse_json(file_path: str) -> List[Dict[str, str]]:
        """
        Parse JSON format chat logs
        Expected format: list of objects with 'speaker', 'text', 'timestamp'

        Args:
            file_path: Path to JSON file

        Returns:
            List of dictionaries with 'speaker', 'text', and 'timestamp'
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("JSON must be a list of message objects")

        messages = []
        for item in data:
            # Support speaker/text format
            if 'speaker' in item and 'text' in item:
                messages.append({
                    'speaker': str(item['speaker']),
                    'text': str(item['text']),
                    'timestamp': str(item.get('timestamp', 'N/A'))
                })
            # Support instruction/response format (e.g. Dolly/Databricks dataset)
            elif 'instruction' in item or 'response' in item:
                instruction = item.get('instruction', '').strip()
                response = item.get('response', '').strip()
                if instruction:
                    messages.append({
                        'speaker': 'User',
                        'text': instruction,
                        'timestamp': 'N/A'
                    })
                if response:
                    messages.append({
                        'speaker': 'AI',
                        'text': response,
                        'timestamp': 'N/A'
                    })

        return messages


class TextPreprocessor:
    """Clean and preprocess text for NLP"""

    _HTML_RE = re.compile(r'<[^>]+>')
    _URL_RE = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    _EMAIL_RE = re.compile(r'\S+@\S+')
    _WHITESPACE_RE = re.compile(r'\s+')
    _SKIP_RE = re.compile(
        r'^Attached \d+ file|^- \s*<a href|^Products:|^Why is this here\?'
    )

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean and normalize text

        Args:
            text: Raw text

        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""

        text = TextPreprocessor._HTML_RE.sub('', text)
        text = TextPreprocessor._URL_RE.sub('', text)
        text = TextPreprocessor._EMAIL_RE.sub('', text)
        text = TextPreprocessor._WHITESPACE_RE.sub(' ', text)
        return text.strip()

    @staticmethod
    def is_valid_message(text: str, min_length: int = 10) -> bool:
        """
        Check if message is valid for sentiment analysis

        Args:
            text: Message text
            min_length: Minimum character length

        Returns:
            True if valid
        """
        if not isinstance(text, str) or len(text) < min_length:
            return False

        if TextPreprocessor._SKIP_RE.match(text):
            return False

        return True


class SentimentAnalyzer:
    """Perform sentiment analysis using transformer models"""

    def __init__(self, model_name: str = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"):
        """
        Initialize sentiment analyzer

        Args:
            model_name: HuggingFace model name
        """
        import torch
        device = 0 if torch.cuda.is_available() else -1
        device_label = "GPU" if device == 0 else "CPU"
        torch_dtype = torch.float16 if device == 0 else torch.float32
        print(f"Loading sentiment analysis model: {model_name} (using {device_label})")
        self.analyzer = pipeline(
            "sentiment-analysis", model=model_name,
            truncation=True, max_length=512,
            device=device, torch_dtype=torch_dtype
        )
        # Compile model for faster inference on PyTorch 2.0+
        try:
            self.analyzer.model = torch.compile(self.analyzer.model)
            print("Model compiled with torch.compile for faster inference.")
        except Exception:
            pass
        # Warm up the model to avoid cold-start latency on first real batch
        self.analyzer("warm up", batch_size=1)
        print("Model loaded successfully!")

    def analyze_message(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of a single message

        Args:
            text: Message text

        Returns:
            Dictionary with 'label' and 'score'
        """
        # Truncate very long messages
        if len(text) > 2000:
            text = text[:2000]

        result = self.analyzer(text)[0]

        # Convert to normalized score (-1 to 1)
        # Positive sentiment: 0 to 1
        # Neutral sentiment: 0
        # Negative sentiment: 0 to -1
        if result['label'] == 'positive':
            score = result['score']
        elif result['label'] == 'negative':
            score = -result['score']
        else:  # neutral
            score = 0.0

        return {
            'label': result['label'],
            'score': score
        }

    def analyze_messages(self, messages: List[Dict[str, str]], batch_size: int = 64) -> pd.DataFrame:
        """
        Analyze sentiment for all messages using batch processing

        Args:
            messages: List of message dictionaries
            batch_size: Number of messages to process at once

        Returns:
            DataFrame with messages and sentiment scores
        """
        # Truncate texts in a single pass (avoids re-slicing later)
        texts = [msg['text'][:2000] for msg in messages]

        print(f"Analyzing sentiment for {len(texts)} messages (batch_size={batch_size})...")
        raw_results = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
            raw_results.extend(self.analyzer(texts[i:i + batch_size]))

        # Build results in a single pass without intermediate dicts
        _score_map = {'positive': 1, 'negative': -1, 'neutral': 0}
        results = [
            {
                'speaker': msg['speaker'],
                'text': msg['text'][:100] + '...' if len(msg['text']) > 100 else msg['text'],
                'full_text': msg['text'],
                'timestamp': msg.get('timestamp', 'N/A'),
                'datetime_parsed': parse_timestamp(msg.get('timestamp', 'N/A')),
                'sentiment_label': s['label'],
                'sentiment_score': s['score'] * _score_map.get(s['label'], 0),
            }
            for msg, s in zip(messages, raw_results)
        ]

        return pd.DataFrame(results)


class SentimentVisualizer:
    """Create visualizations for sentiment analysis results"""

    @staticmethod
    def plot_sentiment_comparison(df: pd.DataFrame, output_path: str = None):
        """
        Create bar chart comparing sentiment between AI and User

        Args:
            df: DataFrame with sentiment analysis results
            output_path: Path to save the plot (optional)
        """
        # Group by speaker and calculate statistics
        speaker_stats = df.groupby('speaker').agg({
            'sentiment_score': ['mean', 'std', 'count']
        }).round(3)

        speaker_stats.columns = ['mean_score', 'std_score', 'message_count']
        speaker_stats = speaker_stats.reset_index()
        speaker_stats['std_score'] = speaker_stats['std_score'].fillna(0)

        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Plot 1: Average Sentiment Score
        colors = ['#3b82f6' if speaker == 'User' else '#10b981' for speaker in speaker_stats['speaker']]
        bars1 = ax1.bar(speaker_stats['speaker'], speaker_stats['mean_score'],
                       yerr=speaker_stats['std_score'], capsize=5, color=colors, alpha=0.8)

        ax1.set_ylabel('Average Sentiment Score', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Speaker', fontsize=12, fontweight='bold')
        ax1.set_title('Average Sentiment: AI vs User', fontsize=14, fontweight='bold')
        ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax1.set_ylim(-1, 1)
        ax1.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom' if height > 0 else 'top',
                    fontweight='bold')

        # Plot 2: Message Count
        bars2 = ax2.bar(speaker_stats['speaker'], speaker_stats['message_count'],
                       color=colors, alpha=0.8)

        ax2.set_ylabel('Number of Messages', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Speaker', fontsize=12, fontweight='bold')
        ax2.set_title('Message Count: AI vs User', fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom',
                    fontweight='bold')

        plt.tight_layout()

        # Save or show
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"\nVisualization saved to: {output_path}")
        else:
            plt.savefig('sentiment_analysis.png', dpi=300, bbox_inches='tight')
            print(f"\nVisualization saved to: sentiment_analysis.png")

        plt.show()

        # Print summary statistics
        print("\n" + "="*60)
        print("SENTIMENT ANALYSIS SUMMARY")
        print("="*60)
        for _, row in speaker_stats.iterrows():
            print(f"\n{row['speaker']}:")
            print(f"  Average Sentiment: {row['mean_score']:.3f} (±{row['std_score']:.3f})")
            print(f"  Total Messages: {int(row['message_count'])}")

            # Interpretation
            if row['mean_score'] > 0.5:
                interpretation = "Very Positive"
            elif row['mean_score'] > 0:
                interpretation = "Slightly Positive"
            elif row['mean_score'] > -0.5:
                interpretation = "Slightly Negative"
            else:
                interpretation = "Very Negative"
            print(f"  Interpretation: {interpretation}")
        print("="*60)

    @staticmethod
    def plot_sentiment_over_time(df: pd.DataFrame, output_path: str = None):
        """
        Plot sentiment scores over time (monthly if real timestamps exist,
        otherwise by equal-sized message-sequence buckets).

        Args:
            df: DataFrame with sentiment analysis results, must contain
                'datetime_parsed', 'speaker', and 'sentiment_score' columns
            output_path: Path to save the plot (optional)
        """
        has_real_ts = df['datetime_parsed'].notna().any()

        speakers = sorted(df['speaker'].unique())
        colors = {'User': '#3b82f6', 'AI': '#10b981'}

        if has_real_ts:
            # ---- Real-timestamp path ----------------------------------------
            df_ts = df[df['datetime_parsed'].notna()].copy()
            df_ts['period'] = df_ts['datetime_parsed'].dt.to_period('M')

            # Determine granularity based on date range
            date_range = (df_ts['datetime_parsed'].max() - df_ts['datetime_parsed'].min()).days
            if date_range <= 60:
                df_ts['period'] = df_ts['datetime_parsed'].dt.to_period('W')
                x_label = 'Week'
            elif date_range <= 730:
                df_ts['period'] = df_ts['datetime_parsed'].dt.to_period('M')
                x_label = 'Month'
            else:
                df_ts['period'] = df_ts['datetime_parsed'].dt.to_period('Q')
                x_label = 'Quarter'

            grouped = (
                df_ts.groupby(['period', 'speaker'])['sentiment_score']
                .mean()
                .reset_index()
            )
            grouped['period_dt'] = grouped['period'].dt.to_timestamp()

            fig, ax = plt.subplots(figsize=(14, 6))

            for speaker in speakers:
                sub = grouped[grouped['speaker'] == speaker].sort_values('period_dt')
                if sub.empty:
                    continue
                color = colors.get(speaker, '#6366f1')
                ax.plot(sub['period_dt'], sub['sentiment_score'],
                        marker='o', label=speaker, color=color, linewidth=2, markersize=5)
                ax.fill_between(sub['period_dt'], sub['sentiment_score'],
                                alpha=0.12, color=color)

            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            fig.autofmt_xdate()
            ax.set_xlabel(x_label, fontsize=12, fontweight='bold')
            title_suffix = f"per {x_label}"
            coverage = (
                f"{df_ts['datetime_parsed'].min().strftime('%b %Y')} – "
                f"{df_ts['datetime_parsed'].max().strftime('%b %Y')}"
            )
            n_without_ts = df['datetime_parsed'].isna().sum()
            if n_without_ts:
                print(f"  Note: {n_without_ts} messages had no timestamp and are excluded from the time-series plot.")

        else:
            # ---- Index-based sequence path ----------------------------------
            n = len(df)
            n_buckets = max(5, min(30, n // max(1, n // 20)))
            df = df.copy()
            df['bucket'] = pd.cut(df.index, bins=n_buckets, labels=False)

            grouped = (
                df.groupby(['bucket', 'speaker'])['sentiment_score']
                .mean()
                .reset_index()
            )

            fig, ax = plt.subplots(figsize=(14, 6))

            for speaker in speakers:
                sub = grouped[grouped['speaker'] == speaker].sort_values('bucket')
                if sub.empty:
                    continue
                color = colors.get(speaker, '#6366f1')
                ax.plot(sub['bucket'], sub['sentiment_score'],
                        marker='o', label=speaker, color=color, linewidth=2, markersize=5)
                ax.fill_between(sub['bucket'], sub['sentiment_score'],
                                alpha=0.12, color=color)

            ax.set_xlabel('Message Sequence (groups of ~equal size)', fontsize=12, fontweight='bold')
            title_suffix = "over Message Sequence"
            coverage = f"{n} messages, {n_buckets} groups"
            print("  Note: No timestamps found in data — using message order as the time axis.")

        ax.set_ylabel('Average Sentiment Score', fontsize=12, fontweight='bold')
        ax.set_title(f'Sentiment {title_suffix}\n({coverage})', fontsize=14, fontweight='bold')
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_ylim(-1, 1)
        ax.grid(axis='y', alpha=0.3)
        ax.legend(fontsize=11)

        plt.tight_layout()

        if output_path:
            stem = Path(output_path).stem
            suffix = Path(output_path).suffix or '.png'
            ts_path = str(Path(output_path).parent / f"{stem}_over_time{suffix}")
        else:
            ts_path = 'sentiment_over_time.png'

        plt.savefig(ts_path, dpi=300, bbox_inches='tight')
        print(f"Time-series visualization saved to: {ts_path}")
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Analyze sentiment in LLM chat logs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python sentiment_analysis.py MyActivity.html
  python sentiment_analysis.py chats.csv -o results.png
  python sentiment_analysis.py conversation.json --export results.csv
        """
    )

    parser.add_argument('input_file', type=str, help='Input file (HTML, CSV, or JSON)')
    parser.add_argument('-o', '--output', type=str, help='Output path for visualization')
    parser.add_argument('--export', type=str, help='Export detailed results to CSV')
    parser.add_argument('--min-length', type=int, default=10,
                       help='Minimum message length to analyze (default: 10)')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for sentiment analysis (default: 64, increase for GPU)')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of records to process (useful for quick testing)')
    parser.add_argument('--no-time-series', action='store_true',
                       help='Skip the sentiment-over-time plot')

    args = parser.parse_args()

    # Validate input file
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: File not found: {args.input_file}")
        return

    # Determine file type and parse
    file_ext = input_path.suffix.lower()
    print(f"\nParsing {file_ext} file: {args.input_file}")

    try:
        if file_ext == '.html':
            messages = ChatLogParser.parse_html(args.input_file)
        elif file_ext == '.csv':
            messages = ChatLogParser.parse_csv(args.input_file)
        elif file_ext == '.json':
            messages = ChatLogParser.parse_json(args.input_file)
        elif file_ext == '.jsonl':
            messages = ChatLogParser.parse_jsonl(args.input_file, max_records=args.limit)
        else:
            print(f"Error: Unsupported file type: {file_ext}")
            print("Supported types: .html, .csv, .json, .jsonl")
            return
    except Exception as e:
        print(f"Error parsing file: {e}")
        return

    if args.limit is not None and file_ext != '.jsonl':
        messages = messages[:args.limit * 2]  # *2 to account for User+AI pairs

    print(f"Found {len(messages)} messages")

    # Preprocess messages in parallel
    print("\nPreprocessing messages...")
    preprocessor = TextPreprocessor()

    def _process(msg):
        cleaned = preprocessor.clean_text(msg['text'])
        if preprocessor.is_valid_message(cleaned, min_length=args.min_length):
            return {**msg, 'text': cleaned}
        return None

    workers = min(8, (len(messages) // 100) or 1)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        valid_messages = [m for m in executor.map(_process, messages) if m is not None]

    print(f"Valid messages after preprocessing: {len(valid_messages)}")

    if len(valid_messages) == 0:
        print("Error: No valid messages found after preprocessing")
        return

    # Perform sentiment analysis
    analyzer = SentimentAnalyzer()
    results_df = analyzer.analyze_messages(valid_messages, batch_size=args.batch_size)

    # Export detailed results if requested
    if args.export:
        results_df.to_csv(args.export, index=False)
        print(f"\nDetailed results exported to: {args.export}")

    # Create visualization
    visualizer = SentimentVisualizer()
    visualizer.plot_sentiment_comparison(results_df, output_path=args.output)

    if not args.no_time_series:
        print("\nGenerating sentiment-over-time plot...")
        visualizer.plot_sentiment_over_time(results_df, output_path=args.output)


if __name__ == "__main__":
    main()

