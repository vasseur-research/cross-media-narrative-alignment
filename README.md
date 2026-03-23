# cross-media-narrative-alignment
Multilingual NLP pipeline (NER, BERTopic, transformer embeddings, Granger causality) tracing narrative alignment across Russian and Colombian news coverage (2013–2023). Identifies temporal coupling in geopolitical framing around the 2022 Russia-Ukraine conflict using 38,000+ Spanish- and Russian-language articles. This repository accompanies the paper *"Tracing Narrative Alignment Over Time Across Media Ecosystems: A Multilingual Computational Approach to Russian and Colombian Coverage"* and provides tools for named entity recognition, sentiment analysis, topic modeling, and temporal cross-series statistical analysis over a multilingual document corpus.

## Repository Structure

```
/
├── config.py                          # Centralized configuration (paths, models, parameters)
├── requirements.txt                   # Python dependencies
├── thread_NER.py                      # Named entity recognition pipeline
├── thread_sentiment.py                # Sentiment analysis pipeline
├── Topic_Cluster.py                   # BERTopic topic modeling (agglomerative + HDBSCAN)
├── ollama_topics.py                   # LLM-based topic label generation via Ollama
├── train_TopicModel.py                # BERTopic training with Llama 2 label generation
└── narrative_alignment_analysis.py    # Temporal alignment, Granger causality, and FDR correction
```

## Dependencies and Installation

Requires Python 3.9+ and a local MongoDB instance.

```bash
pip install -r requirements.txt
```

**Additional infrastructure:**

- **MongoDB** (≥ 4.4): Used as the primary document store. Scripts read from and write results back to a configurable database and collection.
- **Ollama**: Required by `ollama_topics.py` for local LLM inference. Install from [ollama.com](https://ollama.com) and pull the model specified in `config.py` (default: `nous-hermes2`).
- **HuggingFace token** (optional): Required only by `train_TopicModel.py` for gated Llama 2 model access. Set the `HUGGINGFACE_TOKEN` environment variable.
- **GPU/MPS**: Topic modeling and transformer inference benefit from GPU acceleration. The pipeline auto-detects Apple MPS or falls back to CPU.

## Configuration

All paths, model identifiers, database credentials, and tunable hyperparameters are centralized in `config.py`. Values can be overridden via environment variables:

```bash
export MONGODB_URI="mongodb://127.0.0.1:27017/?directConnection=true"
export MONGO_DB_NAME="NMR"
export MONGO_COLLECTION_NAME="RUS-COL"
export OLLAMA_MODEL="nous-hermes2"
```

Key model defaults:

| Parameter | Value |
|---|---|
| Sentence embeddings | `intfloat/multilingual-e5-large-instruct` |
| NER model | `Babelscape/wikineural-multilingual-ner` |
| Sentiment model | `nlptown/bert-base-multilingual-uncased-sentiment` |
| Topic label LLM | `meta-llama/Llama-2-13b-chat-hf` |

## Pipeline Overview

The scripts form a sequential enrichment pipeline over a MongoDB-backed document corpus. Each stage reads from the database, performs inference, and writes structured results back.

### 1. Named Entity Recognition — `thread_NER.py`

Runs the WikiNEuRal multilingual NER model over document texts to extract location (LOC), person (PER), organization (ORG), and miscellaneous (MIS) entities. Location entities are converted to ISO 3166-1 alpha-3 trigraph codes using `country_converter`, enabling geopolitical filtering in downstream analysis. Inference is parallelized across CPU cores via `torch.multiprocessing`.

**MongoDB fields written:** `NER_Country`, `NER_Location`, `NER_Person`, `NER_Organization`, `NER_Misc`

```bash
python thread_NER.py
```

### 2. Sentiment Analysis — `thread_sentiment.py`

Applies a multilingual BERT sentiment classifier to each document. Long texts are split into 512-token chunks with CLS/SEP framing, scored independently, and averaged via softmax pooling to produce a single 1–5 sentiment rating. Like the NER step, inference is parallelized across available CPU cores. Only documents without existing sentiment scores are processed.

**MongoDB field written:** `Sentiment`

```bash
python thread_sentiment.py
```

### 3. Topic Modeling — `Topic_Cluster.py`

Fits BERTopic models to document summaries for each (country, NER target) combination. Two clustering backends are available:

- **Agglomerative clustering** (default path): Selects *k* via the KMeans elbow method on the Calinski-Harabasz index, then fits Agglomerative Clustering with complete linkage.
- **HDBSCAN**: Density-based clustering with configurable `min_cluster_size`.

Both paths use UMAP for dimensionality reduction and extract topic representations via KeyBERT-inspired and Maximal Marginal Relevance methods. Outputs include serialized BERTopic models, document-topic archive CSVs, KeyBERT label files, and hierarchical dendrograms. Topic assignments are written back to MongoDB.

```bash
python Topic_Cluster.py
```

**Output directories** (configurable in `config.py`):
- `data/models/topic_models/` — serialized BERTopic models
- `data/processed/topic_archives/` — document-level topic assignment CSVs
- `data/processed/keybert/` — per-topic KeyBERT keyword files
- `output/dendrograms/` — hierarchical clustering visualizations

### 4. Topic Label Generation — `ollama_topics.py`

A post-processing step that reads KeyBERT label files produced by `Topic_Cluster.py` and prompts a local Ollama model to generate human-readable topic labels. For each topic, representative documents and keywords are sent to the LLM with a few-shot prompt requesting three candidate labels.

```bash
python ollama_topics.py
```

Input files must follow the naming convention `Labels_*.csv` in the KeyBERT labels directory. Output files are prefixed with `REVIEW_`.

### 5. Topic Modeling with Llama 2 Labels — `train_TopicModel.py`

An alternative topic modeling pipeline that replaces the Ollama labeling step with on-device Llama 2 inference via HuggingFace Transformers. Uses HDBSCAN clustering and generates interactive document-topic visualizations with 2D UMAP projections.

```bash
export HUGGINGFACE_TOKEN="hf_..."
python train_TopicModel.py
```

### 6. Temporal Narrative Alignment Analysis — `narrative_alignment_analysis.py`

The main statistical analysis script. Operates on a preprocessed article-level CSV (not MongoDB) and computes, for each combination of year, geopolitical target (trigraph), and macro-narrative label:

- **Weekly aggregation**: Article frequency, summed sentiment, and mean sentiment per country, with min-max and signed scaling.
- **OLS time trends**: Linear trend slope, *R*², and *p*-value for each country's weekly series.
- **Lagged cross-correlation**: Searches lags ±4 weeks for the shift maximizing the absolute Pearson *r* between the Russia and Colombia series.
- **Granger causality**: Likelihood-ratio and *F*-tests at lags 1–4 for the Russia → Colombia direction.
- **Benjamini-Hochberg FDR correction**: Applied across all *p*-values for trend, correlation, and Granger tests.

```bash
# Basic usage
python narrative_alignment_analysis.py \
    --input data/processed/articles.csv

# Full parameterization
python narrative_alignment_analysis.py \
    --input data/processed/articles.csv \
    --output results/weekly_narrative_results_bh.csv \
    --macros "security and conflict" "diplomacy" "economy" "politics and society" \
    --trigraphs USA UKR RUS \
    --years 2022 2023 \
    --max-ccf-lag 4 \
    --max-granger-lag 4
```

**Expected input CSV columns:**

| Column | Type | Description |
|---|---|---|
| `date_adj` | datetime | Publication date |
| `country` | str | `"Colombia"` or `"Russia"` |
| `Sentiment_adj` | int | Sentiment on a [-2, 2] scale |
| `Macro` | list (str) | Macro-narrative labels (may be a stringified Python list) |
| `NER_Trigraph` | list (str) | ISO alpha-3 country codes (may be a stringified Python list) |

**Output:** A CSV with one row per (year, trigraph, macro-narrative, metric) combination, including OLS statistics, best-lag correlation, Granger test results, and BH-corrected *p*-values.

## Data Requirements

- The NER, sentiment, and topic modeling scripts (`thread_NER.py`, `thread_sentiment.py`, `Topic_Cluster.py`, `train_TopicModel.py`) require a running MongoDB instance populated with documents containing at minimum a `body` text field and a `country` field.
- `ollama_topics.py` requires KeyBERT label CSVs produced by `Topic_Cluster.py`.
- `narrative_alignment_analysis.py` requires a pre-processed CSV with the columns described above. If starting from MongoDB, export and preprocess as described in the script's docstring (e.g., remap 1–5 sentiment to the [-2, 2] scale).

## Citation

If you use this pipeline in your research, please cite the accompanying paper:

> *Tracing Narrative Alignment Over Time Across Media Ecosystems: A Multilingual Computational Approach to Russian and Colombian Coverage*

## License

See repository root for license information.
