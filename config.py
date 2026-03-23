"""
config.py
=========
Centralized configuration for the Russia-Colombia narrative analysis pipeline.

All file paths, database credentials, model names, and tunable parameters are
defined here so that individual scripts do not contain hardcoded values.

To adapt the pipeline to your environment:
  1. Copy this file and edit paths/credentials as needed, OR
  2. Set the corresponding environment variables (see defaults below).
"""

import os

# =============================================================================
# Database configuration
# =============================================================================

MONGODB_URI = os.environ.get(
    "MONGODB_URI",
    "mongodb://127.0.0.1:27017/?directConnection=true&serverSelectionTimeoutMS=2000"
)
MONGO_DB_NAME = os.environ.get("MONGO_DB_NAME")
MONGO_COLLECTION_NAME = os.environ.get("MONGO_COLLECTION_NAME")

# =============================================================================
# HuggingFace / API tokens
# =============================================================================

HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN", "")

# =============================================================================
# File paths — override via environment variables or edit defaults
# =============================================================================

# Directory containing raw HTML files downloaded from mid.ru
RUS_ARTICLES_DIR = os.environ.get(
    "RUS_ARTICLES_DIR",
    os.path.join("data", "raw", "RUS_Articles")
)

# Output CSV for parsed Russian MFA corpus
RUS_CORPUS_CSV = os.environ.get(
    "RUS_CORPUS_CSV",
    os.path.join("data", "processed", "corpus_RUS.csv")
)

# Directory for Ollama summary outputs
OLLAMA_SUMMARIES_DIR = os.environ.get(
    "OLLAMA_SUMMARIES_DIR",
    os.path.join("data", "processed", "ollama_summaries")
)

# Directory for KeyBERT label files (input/output for topic labeling)
KEYBERT_LABELS_DIR = os.environ.get(
    "KEYBERT_LABELS_DIR",
    os.path.join("data", "processed", "keybert")
)

# Directory for saved BERTopic models
TOPIC_MODELS_DIR = os.environ.get(
    "TOPIC_MODELS_DIR",
    os.path.join("data", "models", "topic_models")
)

# Directory for topic modeling archive CSVs
TOPIC_ARCHIVE_DIR = os.environ.get(
    "TOPIC_ARCHIVE_DIR",
    os.path.join("data", "processed", "topic_archives")
)

# Directory for dendrograms and label visualizations
DENDROGRAM_DIR = os.environ.get(
    "DENDROGRAM_DIR",
    os.path.join("output", "dendrograms")
)

# =============================================================================
# Model parameters
# =============================================================================

# Ollama model used for summarization and topic labeling
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "nous-hermes2")

# Sentence embedding model for BERTopic
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large-instruct"

# NER model
NER_MODEL_NAME = "Babelscape/wikineural-multilingual-ner"

# Sentiment model
SENTIMENT_MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"

# LLM for topic label generation (used in train_TopicModel.py)
LLAMA_MODEL_ID = "meta-llama/Llama-2-13b-chat-hf"

# =============================================================================
# Scraping parameters
# =============================================================================

# Number of Google search result pages to scrape
SCRAPE_NUM_PAGES = 10

# Colombian Cancilleria base URL
CANCILLERIA_BASE_URL = "https://www.cancilleria.gov.co"

# Max articles to scrape per Cancilleria session
CANCILLERIA_MAX_ARTICLES = 10

# HTTP retry configuration
HTTP_MAX_RETRIES = 5
HTTP_BACKOFF_FACTOR = 0.1

# =============================================================================
# NER / filtering parameters
# =============================================================================

# Country trigraphs used to filter documents by geopolitical relevance
NER_TARGET_TRIGRAPHS = ["RUS", "UKR", "USA"]

# Countries of origin in the corpus
COUNTRIES_OF_ORIGIN = ["Russia", "Colombia"]

# =============================================================================
# Topic modeling parameters
# =============================================================================

# UMAP
UMAP_N_NEIGHBORS = 15
UMAP_N_COMPONENTS = 5
UMAP_MIN_DIST = 0.0
UMAP_METRIC = "cosine"

# HDBSCAN
HDBSCAN_MIN_CLUSTER_SIZE = 15
HDBSCAN_METRIC = "euclidean"
HDBSCAN_CLUSTER_SELECTION_METHOD = "eom"

# KMeans elbow range
KMEANS_K_RANGE = (2, 20)

# Minimum corpus size to attempt topic modeling
MIN_CORPUS_SIZE = 30

# =============================================================================
# Sentiment analysis parameters
# =============================================================================

# BERT token chunk size (including CLS/SEP tokens)
SENTIMENT_CHUNK_SIZE = 512

# =============================================================================
# Thread pool / multiprocessing
# =============================================================================

THREAD_POOL_SIZE = 25
