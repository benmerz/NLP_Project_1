#!/usr/bin/env python3
"""
LLM Chat Log Sentiment Analysis
Analyzes sentiment from chat logs (CSV, HTML, or JSON) and visualizes results.
"""

import re
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
from bs4 import BeautifulSoup
import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
from transformers import pipeline
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


class ChatLogParser:
    """Parse chat logs from various file formats"""

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
            content_div = cell.find('div', class_='content-cell mdl-cell--6-col')
            if not content_div:
                continue

            # Get the full content
            full_text = content_div.get_text(separator='\n', strip=True)

            # Split by "Prompted" to separate user message from AI response
            if 'Prompted' in full_text:
                parts = full_text.split('Prompted', 1)
                if len(parts) > 1:
                    # Extract user message (everything before the first timestamp)
                    user_part = parts[1]

                    # Timestamp pattern: "MMM DD, YYYY, HH:MM:SS AM/PM ZONE"
                    timestamp_pattern = r'([A-Z][a-z]{2} \d{1,2}, \d{4}, \d{1,2}:\d{2}:\d{2} [AP]M [A-Z]{3})'
                    timestamp_match = re.search(timestamp_pattern, user_part)

                    if timestamp_match:
                        timestamp = timestamp_match.group(1)
                        # User message is before the timestamp
                        user_text = user_part[:timestamp_match.start()].strip()
                        # AI response is after the timestamp
                        ai_text = user_part[timestamp_match.end():].strip()

                        # Add user message
                        if user_text:
                            messages.append({
                                'speaker': 'User',
                                'text': user_text,
                                'timestamp': timestamp
                            })

                        # Add AI message
                        if ai_text:
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

        messages = []
        for _, row in df.iterrows():
            messages.append({
                'speaker': str(row['speaker']),
                'text': str(row['text']),
                'timestamp': str(row.get('timestamp', 'N/A'))
            })

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
            if 'speaker' not in item or 'text' not in item:
                continue
            messages.append({
                'speaker': str(item['speaker']),
                'text': str(item['text']),
                'timestamp': str(item.get('timestamp', 'N/A'))
            })

        return messages


class TextPreprocessor:
    """Clean and preprocess text for NLP"""

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean and normalize text

        Args:
            text: Raw text

        Returns:
            Cleaned text
        """
        # Remove HTML tags if any remain
        text = re.sub(r'<[^>]+>', '', text)

        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        # Strip leading/trailing whitespace
        text = text.strip()

        return text

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
        if not text or len(text) < min_length:
            return False

        # Skip messages that are just file attachments or metadata
        skip_patterns = [
            r'^Attached \d+ file',
            r'^- \s*<a href',
            r'^Products:',
            r'^Why is this here\?'
        ]

        for pattern in skip_patterns:
            if re.match(pattern, text):
                return False

        return True


class SentimentAnalyzer:
    """Perform sentiment analysis using transformer models"""

    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        """
        Initialize sentiment analyzer

        Args:
            model_name: HuggingFace model name
        """
        print(f"Loading sentiment analysis model: {model_name}")
        self.analyzer = pipeline("sentiment-analysis", model=model_name, truncation=True, max_length=512)
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
        # Negative sentiment: 0 to -1
        if result['label'] == 'POSITIVE':
            score = result['score']
        else:  # NEGATIVE
            score = -result['score']

        return {
            'label': result['label'],
            'score': score
        }

    def analyze_messages(self, messages: List[Dict[str, str]]) -> pd.DataFrame:
        """
        Analyze sentiment for all messages

        Args:
            messages: List of message dictionaries

        Returns:
            DataFrame with messages and sentiment scores
        """
        results = []

        print("Analyzing sentiment for each message...")
        for msg in tqdm(messages, desc="Processing messages"):
            sentiment = self.analyze_message(msg['text'])
            results.append({
                'speaker': msg['speaker'],
                'text': msg['text'][:100] + '...' if len(msg['text']) > 100 else msg['text'],
                'full_text': msg['text'],
                'timestamp': msg.get('timestamp', 'N/A'),
                'sentiment_label': sentiment['label'],
                'sentiment_score': sentiment['score']
            })

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
        else:
            print(f"Error: Unsupported file type: {file_ext}")
            print("Supported types: .html, .csv, .json")
            return
    except Exception as e:
        print(f"Error parsing file: {e}")
        return

    print(f"Found {len(messages)} messages")

    # Preprocess messages
    print("\nPreprocessing messages...")
    preprocessor = TextPreprocessor()
    valid_messages = []

    for msg in messages:
        cleaned_text = preprocessor.clean_text(msg['text'])
        if preprocessor.is_valid_message(cleaned_text, min_length=args.min_length):
            msg['text'] = cleaned_text
            valid_messages.append(msg)

    print(f"Valid messages after preprocessing: {len(valid_messages)}")

    if len(valid_messages) == 0:
        print("Error: No valid messages found after preprocessing")
        return

    # Perform sentiment analysis
    analyzer = SentimentAnalyzer()
    results_df = analyzer.analyze_messages(valid_messages)

    # Export detailed results if requested
    if args.export:
        results_df.to_csv(args.export, index=False)
        print(f"\nDetailed results exported to: {args.export}")

    # Create visualization
    visualizer = SentimentVisualizer()
    visualizer.plot_sentiment_comparison(results_df, output_path=args.output)


if __name__ == "__main__":
    main()
