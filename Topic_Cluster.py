"""
Topic_Cluster.py
================
BERTopic-based topic modeling pipeline for the Russia-Colombia narrative corpus.

Retrieves document summaries from MongoDB, generates sentence embeddings,
and clusters them using either Agglomerative Clustering (with KMeans elbow
heuristic for k selection) or HDBSCAN. Results are saved as CSV archives,
BERTopic model files, and dendrogram visualizations. Topic IDs are also
written back to MongoDB.

This script processes each (country, NER target) combination separately.
"""

import csv
import os
import re
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from bertopic.vectorizers import ClassTfidfTransformer
from bson.objectid import ObjectId
from hdbscan import HDBSCAN
from pymongo import MongoClient
from scipy.cluster import hierarchy as sch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from yellowbrick.cluster import KElbowVisualizer

from config import (
    DENDROGRAM_DIR,
    EMBEDDING_MODEL_NAME,
    HDBSCAN_CLUSTER_SELECTION_METHOD,
    HDBSCAN_METRIC,
    HDBSCAN_MIN_CLUSTER_SIZE,
    KMEANS_K_RANGE,
    MIN_CORPUS_SIZE,
    MONGODB_URI,
    MONGO_COLLECTION_NAME,
    MONGO_DB_NAME,
    TOPIC_ARCHIVE_DIR,
    TOPIC_MODELS_DIR,
    KEYBERT_LABELS_DIR,
    UMAP_METRIC,
    UMAP_MIN_DIST,
    UMAP_N_COMPONENTS,
    UMAP_N_NEIGHBORS,
)

os.environ["TOKENIZERS_PARALLELISM"] = "True"
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def get_collection():
    """Return the MongoDB collection handle."""
    client = MongoClient(MONGODB_URI)
    return client[MONGO_DB_NAME][MONGO_COLLECTION_NAME]


def topics_cluster(corpus):
    """
    Fit a BERTopic model using Agglomerative Clustering.

    Selects k via the KMeans elbow method (Calinski-Harabasz index),
    then uses Agglomerative Clustering with complete linkage.

    Parameters
    ----------
    corpus : array-like of str
        Document texts to cluster.

    Returns
    -------
    BERTopic
        Fitted topic model.
    """
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    umap_model = UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        n_components=UMAP_N_COMPONENTS,
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
    )
    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

    keybert = KeyBERTInspired()
    mmr = MaximalMarginalRelevance(diversity=0.3)
    representation_model = {"KeyBERT": keybert, "MMR": mmr}

    embeddings = embedding_model.encode(
        corpus, show_progress_bar=True, normalize_embeddings=True
    )

    # Determine k via elbow method
    visualizer = KElbowVisualizer(
        KMeans(), k=KMEANS_K_RANGE, metric="calinski_harabasz", timings=True
    )
    visualizer.fit(embeddings)
    k_score = int(visualizer.elbow_score_)
    if k_score > len(corpus):
        k_score = k_score // 2

    cluster_model = AgglomerativeClustering(n_clusters=k_score, linkage="complete")

    topic_model = BERTopic(
        embedding_model=embedding_model,
        top_n_words=10,
        verbose=True,
        representation_model=representation_model,
        ctfidf_model=ctfidf_model,
        umap_model=umap_model,
        hdbscan_model=cluster_model,
    )

    topic_model.fit_transform(corpus, embeddings)
    return topic_model


def topics_hdbscan(corpus):
    """
    Fit a BERTopic model using HDBSCAN for density-based clustering.

    Parameters
    ----------
    corpus : array-like of str
        Document texts to cluster.

    Returns
    -------
    BERTopic
        Fitted topic model.
    """
    umap_model = UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        n_components=UMAP_N_COMPONENTS,
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
    )
    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
    hdbscan_model = HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        metric=HDBSCAN_METRIC,
        cluster_selection_method=HDBSCAN_CLUSTER_SELECTION_METHOD,
        prediction_data=True,
    )
    vectorizer_model = CountVectorizer(
        stop_words=None, ngram_range=(1, 5), max_features=15_000
    )

    keybert = KeyBERTInspired()
    mmr = MaximalMarginalRelevance(diversity=0.5)
    representation_model = {"KeyBERT": keybert, "MMR": mmr}

    topic_model = BERTopic(
        language="multilingual",
        verbose=True,
        representation_model=representation_model,
        ctfidf_model=ctfidf_model,
        umap_model=umap_model,
        vectorizer_model=vectorizer_model,
        hdbscan_model=hdbscan_model,
    )

    topic_model.fit_transform(corpus)
    return topic_model


def save_upload(topic_model, corpus, site, target, items, collection):
    """
    Save model artifacts, CSV archives, and dendrogram; update MongoDB.

    Parameters
    ----------
    topic_model : BERTopic
        Fitted topic model.
    corpus : array-like of str
        Document texts used for fitting.
    site : str
        Country label (e.g., "Russia").
    target : str
        NER trigraph target (e.g., "UKR").
    items : list of tuple
        (document, mongoID, NER_data) triples.
    collection : pymongo.collection.Collection
        MongoDB collection to update.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    os.makedirs(TOPIC_MODELS_DIR, exist_ok=True)
    topic_model.save(os.path.join(TOPIC_MODELS_DIR, f"TopicModel_{site}_{target}_{timestamp}"))

    df_topics = pd.DataFrame(topic_model.get_document_info(corpus))
    df_items = pd.DataFrame(items, columns=["Document", "mongoID", "NERdata"])
    df_cons = pd.merge_ordered(df_topics, df_items, on="Document")

    os.makedirs(TOPIC_ARCHIVE_DIR, exist_ok=True)
    df_cons.to_csv(
        os.path.join(TOPIC_ARCHIVE_DIR, f"df_archive_{site}_{target}_{timestamp}.csv"),
        index=False,
    )

    df_labels = pd.DataFrame(topic_model.get_topic_info())
    df_labels.drop(["Representation", "Name", "MMR"], axis="columns", inplace=True)
    os.makedirs(KEYBERT_LABELS_DIR, exist_ok=True)
    df_labels.to_csv(
        os.path.join(KEYBERT_LABELS_DIR, f"Labels_{site}_{target}_{timestamp}.csv"),
        index=False,
    )

    # Generate hierarchical dendrogram
    try:
        hierarchical_topics = topic_model.hierarchical_topics(corpus)
        linkage_function = lambda x: sch.linkage(x, "single", optimal_ordering=True)
        hierarchical_topics = topic_model.hierarchical_topics(
            corpus, linkage_function=linkage_function
        )
        dendrogram = topic_model.visualize_hierarchy(
            hierarchical_topics=hierarchical_topics
        )
        os.makedirs(DENDROGRAM_DIR, exist_ok=True)
        dendrogram.write_image(
            os.path.join(DENDROGRAM_DIR, f"dendrogram_{site}_{target}.png")
        )
    except Exception:
        print("Dendrogram generation failed — likely insufficient samples.")

    update_mongo(df_cons, site, target, collection)


def update_mongo(df_final, site, target, collection):
    """
    Write topic assignments back to MongoDB.

    Parameters
    ----------
    df_final : pd.DataFrame
        Merged DataFrame with mongoID and Topic columns.
    site : str
        Country label.
    target : str
        NER trigraph target.
    collection : pymongo.collection.Collection
        MongoDB collection to update.
    """
    for mongo_id, topic_id in zip(df_final["mongoID"], df_final["Topic"]):
        collection.find_one_and_update(
            {"_id": ObjectId(mongo_id)},
            {"$set": {f"Topic, {site}_{target}": topic_id}},
        )
    print(f"MongoDB updated for {site}_{target}")


def get_dataset(site, target, collection):
    """
    Retrieve and preprocess documents for topic modeling.

    Parameters
    ----------
    site : str
        Country of origin ("Russia" or "Colombia").
    target : str
        NER trigraph to filter by (e.g., "UKR").
    collection : pymongo.collection.Collection
        Source MongoDB collection.

    Returns
    -------
    tuple of (pd.Series, list of tuple)
        Cleaned document texts and (document, mongoID, NER_data) triples.
    """
    print(f"Processing {site} for target: {target}")

    dataset = pd.DataFrame(
        list(collection.find(
            {},
            {"_id", "NER_Trigraph", "country", "Ollama Summary", f"Topic, {site}_{target}"},
        ))
    )
    dataset = dataset[dataset["country"] == site]

    df = dataset.copy()
    df = df[df["NER_Trigraph"].apply(lambda x: target in x)]
    df["body"] = df["Ollama Summary"]
    df = df[df["body"].notna()].reset_index(drop=True)

    print(f"Full dataset: {len(df)}")

    # Clean text
    df["body"] = df["body"].str.strip()
    df["body"] = df["body"].apply(lambda x: re.sub(r"\n", " ", x))
    df["body"] = df["body"].str.replace(r"[|]", " ", regex=False)
    df["body"] = df["body"].str.replace(r"\s+", " ", regex=True)
    df["body"] = df["body"].astype("str")

    print(f"{len(df)} documents filtered by {site} for target: {target}")

    corpus = df["body"]
    items = list(zip(corpus, df["_id"], df["NER_Trigraph"]))

    return corpus, items


def main():
    """Run topic modeling for all country-target combinations."""
    collection = get_collection()

    countries = {
        "Russia": ["UKR", "USA"],
        "Colombia": ["RUS", "UKR", "USA"],
    }

    for site, targets in countries.items():
        for target in targets:
            corpus, items = get_dataset(site, target, collection)
            if len(corpus) > MIN_CORPUS_SIZE:
                topic_model = topics_cluster(corpus)
                save_upload(topic_model, corpus, site, target, items, collection)
            else:
                print(f"Skipping {site}/{target}: only {len(corpus)} documents (min: {MIN_CORPUS_SIZE})")


if __name__ == "__main__":
    main()
