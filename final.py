#!/usr/bin/env python3
"""
final.py

Lean sentiment-over-time script for chat data.
Design goal: keep the NLP flow recognizable to text2vec.ipynb while producing
User vs AI sentiment trends as an image.
"""

import argparse
import json
import os
import re
from functools import lru_cache
from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from transformers import pipeline

try:
	from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
except ModuleNotFoundError:
	# Fallback so script can run even if scikit-learn is not installed.
	ENGLISH_STOP_WORDS = frozenset({
		"a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he", "in", "is",
		"it", "its", "of", "on", "that", "the", "to", "was", "were", "will", "with", "i", "you", "we",
		"they", "this", "those", "these", "or", "if", "but", "not", "so", "do", "does", "did",
	})

try:
	import gensim.downloader as api
except Exception:
	api = None

# -----------------------------
# Data loading (timeline context)
# -----------------------------

_TS_PARSE_RE = re.compile(
	r"([A-Z][a-z]{2,3}) (\d{1,2}), (\d{4}), (\d{1,2}):(\d{2}):(\d{2})[\s\u202f]+([AP]M)"
)


def log_progress(message: str):
	print(message, flush=True)


def parse_timestamp(ts: str):
	if not ts or ts == "N/A":
		return None
	# Fast path for Google Takeout-style timestamps.
	match = _TS_PARSE_RE.search(ts)
	if match:
		month, day, year, hour, minute, second, ampm = match.groups()
		try:
			return datetime.strptime(
				f"{month} {day} {year} {hour}:{minute}:{second} {ampm}",
				"%b %d %Y %I:%M:%S %p",
			)
		except ValueError:
			pass

	# Broad fallback for ISO/standard timestamp strings.
	try:
		parsed = pd.to_datetime(ts, errors="coerce", utc=False)
		if pd.notna(parsed):
			return parsed.to_pydatetime()
	except Exception:
		pass

	return None


def load_html(path: str):
	ts_html_re = re.compile(
		r"([A-Z][a-z]{2,3} \d{1,2}, \d{4}, \d{1,2}:\d{2}:\d{2}[\s\u202f]+[AP]M [A-Z]{3})"
	)
	with open(path, encoding="utf-8") as f:
		soup = BeautifulSoup(f.read(), "html.parser")

	messages = []
	for cell in soup.find_all("div", class_="outer-cell"):
		content = cell.find(
			"div",
			class_=lambda x: x
			and "content-cell" in x
			and "mdl-cell--6-col" in x
			and "mdl-typography--body-1" in x,
		)
		if not content:
			continue

		children = list(content.children)
		first = str(children[0]) if children else ""
		if "Prompted" not in first:
			continue

		prompted = first.split("Prompted", 1)[1].strip()
		parts = prompted.split("\n\n", 1)
		user_text = parts[1].strip() if len(parts) == 2 else parts[0].strip()

		timestamp = "N/A"
		ai_parts = []
		seen_ts = False
		for child in children:
			s = str(child).strip()
			if not seen_ts:
				m = ts_html_re.search(s)
				if m:
					timestamp = m.group(1)
					seen_ts = True
				continue
			if hasattr(child, "get_text"):
				t = child.get_text(separator=" ", strip=True)
				if t:
					ai_parts.append(t)

		ai_text = " ".join(ai_parts)
		if user_text:
			messages.append({"speaker": "User", "text": user_text, "timestamp": timestamp})
		if ai_text:
			messages.append({"speaker": "AI", "text": ai_text, "timestamp": timestamp})

	return messages


def load_csv(path: str):
	df = pd.read_csv(path)
	if "speaker" not in df.columns or "text" not in df.columns:
		raise ValueError("CSV must include 'speaker' and 'text' columns")
	return [
		{
			"speaker": str(r["speaker"]),
			"text": str(r["text"]),
			"timestamp": str(r.get("timestamp", "N/A")),
		}
		for r in df.to_dict("records")
	]


def load_json(path: str):
	with open(path, encoding="utf-8") as f:
		data = json.load(f)
	if not isinstance(data, list):
		raise ValueError("JSON input must be a list")

	out = []
	for item in data:
		ts = str(item.get("timestamp", "N/A"))
		if "speaker" in item and "text" in item:
			out.append({"speaker": str(item["speaker"]), "text": str(item["text"]), "timestamp": ts})
		elif "instruction" in item or "response" in item:
			if item.get("instruction"):
				out.append({"speaker": "User", "text": item["instruction"].strip(), "timestamp": ts})
			if item.get("response"):
				out.append({"speaker": "AI", "text": item["response"].strip(), "timestamp": ts})
	return out


def load_jsonl(path: str, limit=None):
	out = []
	log_progress(f"[progress] reading JSONL: {path}")
	with open(path, encoding="utf-8") as f:
		for i, line in enumerate(f):
			if limit is not None and i >= limit:
				break
			if not line.strip():
				continue
			item = json.loads(line)
			ts_value = (
				item.get("timestamp")
				or item.get("datetime")
				or item.get("created_at")
				or item.get("time")
				or item.get("date")
				or "N/A"
			)
			if item.get("instruction"):
				out.append({"speaker": "User", "text": item["instruction"].strip(), "timestamp": str(ts_value)})
			if item.get("response"):
				out.append({"speaker": "AI", "text": item["response"].strip(), "timestamp": str(ts_value)})
			if (i + 1) % 2000 == 0:
				log_progress(f"[progress] read {i + 1} JSONL rows")
	log_progress(f"[progress] finished reading JSONL rows: {i + 1 if 'i' in locals() else 0}")
	return out


def load_messages(path: str, limit=None):
	ext = Path(path).suffix.lower()
	if ext == ".html":
		msgs = load_html(path)
	elif ext == ".csv":
		msgs = load_csv(path)
	elif ext == ".json":
		msgs = load_json(path)
	elif ext == ".jsonl":
		msgs = load_jsonl(path, limit=limit)
		return msgs
	else:
		raise ValueError("Supported types: .html, .csv, .json, .jsonl")

	return msgs[: limit * 2] if limit else msgs


# -----------------------------
# Text Cleaning & Preprocessing 
# -----------------------------

# Keep negation terms so phrases like "not bad" are not inverted to "bad".
NEGATION_WORDS = {"not", "no", "nor", "never"}
STOP_WORDS = ENGLISH_STOP_WORDS - NEGATION_WORDS


def clean_preprocess(text):
	text = text.lower()
	text = re.sub(r"http\S+", " ", text)
	text = re.sub(r"<[^>]+>", " ", text)
	tokens = re.findall(r"[a-z']+", text)
	preprocessed_text = [word for word in tokens if word not in STOP_WORDS]
	return preprocessed_text


@lru_cache(maxsize=200000)
def clean_preprocess_cached(text):
	return clean_preprocess(text)


def clean_for_sentiment(text):
	text = text.lower()
	text = re.sub(r"http\S+", " ", text)
	text = re.sub(r"<[^>]+>", " ", text)
	return re.sub(r"\s+", " ", text).strip()


# -----------------------------
# Text Representation / Feature Engineering 
# -----------------------------

def load_pretrained_word2vec_model():
	"""
	Notebook-like pretrained embedding load.
	Uses a smaller model for practical runtime vs google-news-300.
	"""
	if api is None:
		log_progress("[progress] gensim not available, skipping pretrained word vectors")
		return None
	try:
		log_progress("[progress] loading pretrained word vectors (glove-wiki-gigaword-100)...")
		return api.load("glove-wiki-gigaword-100")
	except Exception:
		log_progress("[progress] failed to load pretrained vectors, continuing without them")
		return None


def convert_word_to_vector(token_lists, pretrained_word2vec_model):
	"""
	Notebook-style averaging logic:
	- initialize zero vector per row
	- add vectors for in-vocab tokens
	- average by token count
	- fallback to zero vector when nothing matches
	"""
	if pretrained_word2vec_model is None:
		return [np.array([], dtype=float) for _ in token_lists]

	vector_size = int(getattr(pretrained_word2vec_model, "vector_size", 100))
	total_vector = []
	total_rows = len(token_lists)
	log_progress(f"[progress] converting text to vectors: 0/{total_rows}")

	for idx, row in enumerate(token_lists, start=1):
		count = 0
		row_total_vector = np.zeros(vector_size)
		for word in row:
			if word in pretrained_word2vec_model:
				row_total_vector = row_total_vector + pretrained_word2vec_model[word]
				count = count + 1

		if count != 0:
			row_avg_vector = row_total_vector / count
			total_vector.append(row_avg_vector)
		else:
			total_vector.append(np.zeros(vector_size))

		if idx % 500 == 0 or idx == total_rows:
			log_progress(f"[progress] converting text to vectors: {idx}/{total_rows}")

	return total_vector


# -----------------------------
# Applying model + timeline plot
# -----------------------------

def build_model(model_max_length=128, model_name="cardiffnlp/twitter-roberta-base-sentiment-latest"):
	device = -1
	pipe_kwargs = {}
	try:
		import torch
		torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
		torch.set_grad_enabled(False)
		if torch.cuda.is_available():
			device = 0
			pipe_kwargs["torch_dtype"] = torch.float16
	except Exception:
		pass

	return pipeline(
		"sentiment-analysis",
		model=model_name,
		truncation=True,
		max_length=model_max_length,
		device=device,
		**pipe_kwargs,
	)


def sentiment_from_prediction(pred):
	"""Return (label, score) where score is probability margin: pos - neg."""
	# Pipeline may return one dict or a list of dicts depending on params/model.
	if isinstance(pred, dict):
		label = str(pred.get("label", "")).upper()
		score = float(pred.get("score", 0.0))
		if "NEG" in label or label == "LABEL_0":
			pos, neg, neu = 1.0 - score, score, 0.0
		elif "POS" in label or label == "LABEL_2":
			pos, neg, neu = score, 1.0 - score, 0.0
		elif "NEU" in label or label == "LABEL_1":
			pos, neg, neu = 0.0, 0.0, score
		else:
			pos, neg, neu = score, 1.0 - score, 0.0
	else:
		pos = neg = neu = 0.0
		for item in pred:
			label = str(item.get("label", "")).upper()
			value = float(item.get("score", 0.0))
			if "NEG" in label or label == "LABEL_0":
				neg = max(neg, value)
			elif "POS" in label or label == "LABEL_2":
				pos = max(pos, value)
			elif "NEU" in label or label == "LABEL_1":
				neu = max(neu, value)

	margin = pos - neg
	if pos >= neg and pos >= neu:
		return "POSITIVE", margin
	if neg >= pos and neg >= neu:
		return "NEGATIVE", margin
	return "NEUTRAL", margin


def score_messages(messages, model, batch_size=256, use_word_vectors=False, max_chars=220, neutral_threshold=0.0):
	log_progress(f"[progress] preprocessing messages: {len(messages)} total")
	raw_texts = [m["text"] for m in messages]
	unique_raw_texts = list(dict.fromkeys(raw_texts))
	log_progress(f"[progress] unique raw texts to preprocess: {len(unique_raw_texts)}")
	preprocessed_by_raw = {text: clean_preprocess_cached(text) for text in unique_raw_texts}
	token_lists = [preprocessed_by_raw[text] for text in raw_texts]
	x_features = [clean_for_sentiment(text)[:max_chars] for text in raw_texts]

	# Filter too-short samples after preprocessing.
	filtered = []
	filtered_features = []
	if use_word_vectors:
		pretrained_word2vec_model = load_pretrained_word2vec_model()
		x_vectors = convert_word_to_vector(token_lists, pretrained_word2vec_model)
		for msg, feat, vec in zip(messages, x_features, x_vectors):
			if feat.strip() and (vec.size == 0 or float(np.linalg.norm(vec)) > 0.0):
				filtered.append(msg)
				filtered_features.append(feat)
	else:
		for msg, feat in zip(messages, x_features):
			if feat.strip():
				filtered.append(msg)
				filtered_features.append(feat)

	log_progress(f"[progress] ready for sentiment scoring: {len(filtered_features)} messages")
	if not filtered_features:
		return pd.DataFrame(columns=["speaker", "timestamp", "datetime_parsed", "sentiment_label", "sentiment_score"])

	# Score unique texts once, then map results back to all rows.
	unique_texts = list(dict.fromkeys(filtered_features))
	log_progress(f"[progress] unique texts to score: {len(unique_texts)}")

	raw_unique = []
	total_batches = (len(unique_texts) + batch_size - 1) // batch_size
	for batch_idx in range(total_batches):
		start = batch_idx * batch_size
		end = start + batch_size
		batch = unique_texts[start:end]
		log_progress(f"[progress] running sentiment batch {batch_idx + 1}/{total_batches}")
		raw_unique.extend(model(batch, batch_size=batch_size, top_k=None))
		if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
			log_progress(f"[progress] completed sentiment batch {batch_idx + 1}/{total_batches}")

	pred_by_text = {text: pred for text, pred in zip(unique_texts, raw_unique)}
	raw = [pred_by_text[text] for text in filtered_features]

	rows = []
	for msg, pred in zip(filtered, raw):
		label, score = sentiment_from_prediction(pred)
		rows.append(
			{
				"speaker": msg["speaker"],
				"timestamp": msg["timestamp"],
				"datetime_parsed": parse_timestamp(msg["timestamp"]),
				"sentiment_label": label,
				"sentiment_score": float(score),
			}
		)

	df = pd.DataFrame(rows)
	if neutral_threshold > 0:
		before = len(df)
		df = df[df["sentiment_score"].abs() > neutral_threshold].copy()
		log_progress(f"[progress] removed {before - len(df)} near-neutral rows (|score| <= {neutral_threshold})")
	return df


def plot_sentiment_timeline(df, output_path="sentiment_over_time.png"):
	fig, ax = plt.subplots(figsize=(13, 6))
	colors = {"User": "#2563eb", "AI": "#059669"}

	has_ts = df["datetime_parsed"].notna().any()
	speakers = sorted(df["speaker"].unique())

	if has_ts:
		df2 = df[df["datetime_parsed"].notna()].copy()
		# Always plot by calendar month (e.g., Mar 2025, Apr 2025, ...)
		df2["period"] = df2["datetime_parsed"].dt.to_period("M")
		grouped = df2.groupby(["period", "speaker"], as_index=False)["sentiment_score"].mean()
		grouped["x"] = grouped["period"].dt.to_timestamp()

		for speaker in speakers:
			sub = grouped[grouped["speaker"] == speaker].sort_values("x")
			ax.plot(
				sub["x"],
				sub["sentiment_score"],
				marker="o",
				linewidth=2,
				label=speaker,
				color=colors.get(speaker, "#6b7280"),
			)

		ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
		fig.autofmt_xdate()
		ax.set_xlabel("Month")
		title = "Sentiment Over Time (Monthly)"
	else:
		log_progress("[progress] no parseable datestamps found; using message sequence x-axis")
		# If timestamps are missing, preserve chronological message order in groups.
		df2 = df.copy().reset_index(drop=True)
		n_buckets = max(8, min(30, len(df2) // 20 if len(df2) > 20 else 8))
		df2["bucket"] = pd.cut(df2.index, bins=n_buckets, labels=False, include_lowest=True)
		grouped = df2.groupby(["bucket", "speaker"], as_index=False)["sentiment_score"].mean()

		for speaker in speakers:
			sub = grouped[grouped["speaker"] == speaker].sort_values("bucket")
			ax.plot(
				sub["bucket"],
				sub["sentiment_score"],
				marker="o",
				linewidth=2,
				label=speaker,
				color=colors.get(speaker, "#6b7280"),
			)

		ax.set_xlabel("Message Sequence (grouped)")
		title = "Sentiment Over Message Sequence"

	ax.axhline(0, color="gray", linestyle="--", linewidth=1)
	ax.set_ylabel("Average Sentiment Score")
	ax.set_title(title)
	ax.set_ylim(-1, 1)
	ax.grid(axis="y", alpha=0.25)
	ax.text(
		0.01,
		0.02,
		"Sentiment scale: +1 = very positive, 0 = neutral, -1 = very negative",
		transform=ax.transAxes,
		fontsize=9,
		color="#374151",
		bbox={"boxstyle": "round,pad=0.3", "facecolor": "#f3f4f6", "alpha": 0.9, "edgecolor": "#d1d5db"},
	)
	ax.legend()
	plt.tight_layout()
	plt.savefig(output_path, dpi=300, bbox_inches="tight")
	print(f"Saved image: {output_path}")


def main():
	parser = argparse.ArgumentParser(
		description="Generate User vs AI sentiment-over-time chart from chat data"
	)
	parser.add_argument("input_file", help="Path to .html, .csv, .json, or .jsonl data file")
	parser.add_argument("-o", "--output", default="sentiment_over_time.png", help="Output image path")
	parser.add_argument("--model", default="cardiffnlp/twitter-roberta-base-sentiment-latest", help="HF sentiment model name")
	parser.add_argument("--batch-size", type=int, default=256, help="Inference batch size")
	parser.add_argument("--model-max-length", type=int, default=128, help="Tokenizer max length (smaller = faster)")
	parser.add_argument("--max-chars", type=int, default=220, help="Max characters per cleaned text (smaller = faster)")
	parser.add_argument("--limit", type=int, default=None, help="Optional max records to load")
	parser.add_argument("--use-word-vectors", action="store_true", help="Enable notebook-style word2vec averaging (slower)")
	parser.add_argument("--neutral-threshold", type=float, default=0.0, help="Drop near-neutral scores where |score| <= threshold")
	parser.add_argument("--ultra-fast", action="store_true", help="Aggressive speed mode")
	args = parser.parse_args()

	if args.ultra_fast:
		args.batch_size = max(args.batch_size, 512)
		args.model_max_length = min(args.model_max_length, 96)
		args.max_chars = min(args.max_chars, 160)

	input_path = Path(args.input_file)
	if not input_path.exists():
		raise FileNotFoundError(f"Input file not found: {args.input_file}")

	log_progress(f"[progress] loading messages from: {args.input_file}")
	messages = load_messages(str(input_path), limit=args.limit)
	log_progress(f"[progress] loaded {len(messages)} raw messages")
	if not messages:
		raise ValueError("No messages found in input file.")

	log_progress("[progress] loading sentiment model...")
	model = build_model(model_max_length=args.model_max_length, model_name=args.model)
	log_progress("[progress] scoring sentiment...")
	scored_df = score_messages(
		messages,
		model=model,
		batch_size=args.batch_size,
		use_word_vectors=args.use_word_vectors,
		max_chars=args.max_chars,
		neutral_threshold=args.neutral_threshold,
	)

	if scored_df.empty:
		raise ValueError("No messages remained after preprocessing.")

	log_progress("[progress] generating timeline plot...")
	plot_sentiment_timeline(scored_df, output_path=args.output)
	log_progress("[progress] done")


if __name__ == "__main__":
	main()
