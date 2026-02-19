# LLM Chat Log Sentiment Analysis

A Python script that performs sentiment analysis on chat logs from LLM conversations (Gemini, Bard, ChatGPT, etc.) and visualizes the results.

## Features

- **Multi-format Support**: Parses CSV, HTML (Google Takeout), and JSON files
- **Text Preprocessing**: Cleans and prepares text for NLP analysis
- **BERT-based Sentiment Analysis**: Uses state-of-the-art transformer models
- **Visual Comparison**: Creates bar charts comparing sentiment between AI and User
- **Export Results**: Optionally export detailed analysis to CSV

## Installation

1. Clone or download this repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python sentiment_analysis.py MyActivity.html
```

### With Custom Output Path

```bash
python sentiment_analysis.py MyActivity.html -o my_results.png
```

### Export Detailed Results

```bash
python sentiment_analysis.py chats.csv --export detailed_results.csv
```

### Command Line Options

- `input_file`: Path to your chat log file (required)
- `-o, --output`: Custom output path for the visualization (default: sentiment_analysis.png)
- `--export`: Export detailed results to CSV file
- `--min-length`: Minimum message length to analyze (default: 10 characters)

## Supported File Formats

### 1. HTML (Google Takeout - Gemini Apps)

Export your Gemini chat history using [Google Takeout](https://takeout.google.com/):
1. Go to Google Takeout
2. Deselect all
3. Find and select "My Activity"
4. Click "All activity data included" and select only "Gemini Apps"
5. Download and extract the MyActivity.html file

### 2. CSV Format

CSV file must contain the following columns:
- `speaker`: Either "User" or "AI" (or similar)
- `text`: The message content
- `timestamp`: (optional) Timestamp of the message

Example:
```csv
speaker,text,timestamp
User,"How do I analyze sentiment?","2026-01-15 10:30:00"
AI,"I can help you with that using NLP techniques...","2026-01-15 10:30:05"
```

### 3. JSON Format

JSON file should be a list of message objects:

```json
[
  {
    "speaker": "User",
    "text": "How do I analyze sentiment?",
    "timestamp": "2026-01-15 10:30:00"
  },
  {
    "speaker": "AI",
    "text": "I can help you with that using NLP techniques...",
    "timestamp": "2026-01-15 10:30:05"
  }
]
```

## How It Works

1. **Parsing**: Extracts messages from your chat log file
2. **Preprocessing**: Cleans text by removing URLs, extra whitespace, and HTML tags
3. **Sentiment Analysis**: Uses DistilBERT (fine-tuned on SST-2) to analyze each message
4. **Aggregation**: Calculates average sentiment scores for User vs AI
5. **Visualization**: Creates bar charts showing sentiment comparison

## Sentiment Score Interpretation

- **Score Range**: -1 (very negative) to +1 (very positive)
- **> 0.5**: Very Positive
- **0 to 0.5**: Slightly Positive
- **-0.5 to 0**: Slightly Negative
- **< -0.5**: Very Negative

## Output

The script generates:
1. **Console Output**: Summary statistics and interpretation
2. **Visualization**: Two bar charts showing:
   - Average sentiment score (with error bars)
   - Message count for each speaker
3. **Optional CSV Export**: Detailed per-message analysis

## Example Output

```
SENTIMENT ANALYSIS SUMMARY
============================================================

User:
  Average Sentiment: 0.324 (±0.156)
  Total Messages: 45
  Interpretation: Slightly Positive

AI:
  Average Sentiment: 0.687 (±0.112)
  Total Messages: 45
  Interpretation: Very Positive
============================================================
```

## Requirements

- Python 3.8 or higher
- 2GB+ RAM (for loading transformer models)
- Internet connection (first run only, to download the model)

## Model Information

This script uses `distilbert-base-uncased-finetuned-sst-2-english` from Hugging Face, which is:
- A lightweight BERT variant
- Fine-tuned on the Stanford Sentiment Treebank
- Optimized for sentiment classification
- ~67M parameters

## Troubleshooting

### "Out of Memory" Error
- Try analyzing smaller batches
- Use a machine with more RAM
- Consider using a lighter model

### "No valid messages found"
- Check your file format matches the expected structure
- Try lowering `--min-length`
- Verify the file contains actual chat content

### Slow Processing
- The first run downloads the ML model (~250MB)
- Subsequent runs are faster
- Large files take longer to process

## License

MIT License - Feel free to use and modify as needed

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

## Author

Created for NLP_Project_1