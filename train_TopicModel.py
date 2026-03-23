"""
train_TopicModel.py
===================
BERTopic training with Llama 2 representation model for topic label
generation.

Retrieves documents from MongoDB, computes sentence embeddings, clusters
them with HDBSCAN, and uses a Llama 2 model (via HuggingFace Transformers)
to generate human-readable topic labels. Produces interactive topic
visualizations.

NOTE: Requires a valid HuggingFace token with access to the Llama 2 model.
      Set the HUGGINGFACE_TOKEN environment variable before running.
"""

import torch
import pandas as pd
from bertopic import BERTopic
from bertopic.representation import (
    KeyBERTInspired,
    MaximalMarginalRelevance,
    TextGeneration,
)
from hdbscan import HDBSCAN
from huggingface_hub import login
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline as hf_pipeline
from umap import UMAP

from config import (
    HUGGINGFACE_TOKEN,
    LLAMA_MODEL_ID,
    MONGODB_URI,
    MONGO_DB_NAME,
    UMAP_METRIC,
    UMAP_MIN_DIST,
    UMAP_N_COMPONENTS,
    UMAP_N_NEIGHBORS,
)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Llama 2 prompt templates for topic labeling
SYSTEM_PROMPT = """
<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant for labeling topics.
<</SYS>>
"""

EXAMPLE_PROMPT = """
I have a topic that contains the following documents:
- Traditional diets in most cultures were primarily plant-based with a little meat on top, but with the rise of industrial style meat production and factory farming, meat has become a staple food.
- Meat, but especially beef, is the word food in terms of emissions.
- Eating meat doesn't make you a bad person, not eating meat doesn't make you a good one.

The topic is described by the following keywords: 'meat, beef, eat, eating, emissions, steak, food, health, processed, chicken'.

Based on the information about the topic above, please create a short label of this topic. Make sure you to only return the label and nothing more.

[/INST] Environmental impacts of eating meat
"""

MAIN_PROMPT = """
[INST]
I have a topic that contains the following documents:
[DOCUMENTS]

The topic is described by the following keywords: '[KEYWORDS]'.

Based on the information about the topic above, please create a short label of this topic. Make sure you to only return the label and nothing more.
[/INST]
"""

FULL_PROMPT = SYSTEM_PROMPT + EXAMPLE_PROMPT + MAIN_PROMPT


def run_topic_model(items):
    """
    Train a BERTopic model with Llama 2 topic labeling.

    Parameters
    ----------
    items : tuple
        (docs, titles, country_ids, ner_data) where each is an array-like.
    """
    docs, titles, country_ids, ner_data = items

    # Authenticate with HuggingFace
    if not HUGGINGFACE_TOKEN:
        raise ValueError(
            "HUGGINGFACE_TOKEN is not set. Export it as an environment variable "
            "or set it in config.py."
        )
    login(token=HUGGINGFACE_TOKEN)

    tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        LLAMA_MODEL_ID, device_map="auto", trust_remote_code=False, load_in_8bit=False
    )
    model.eval()

    generator = hf_pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        temperature=0.1,
        max_new_tokens=500,
        repetition_penalty=1.1,
    )

    # Embeddings
    embedding_model = SentenceTransformer("BAAI/bge-large-en")
    embeddings = embedding_model.encode(docs, show_progress_bar=True)

    # Dimensionality reduction
    umap_model = UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        n_components=UMAP_N_COMPONENTS,
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
        random_state=42,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=150,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    reduced_embeddings = UMAP(
        n_neighbors=15, n_components=2, min_dist=0.0,
        metric="cosine", random_state=42,
    ).fit_transform(embeddings)

    # Representation models
    keybert = KeyBERTInspired()
    mmr = MaximalMarginalRelevance(diversity=0.3)
    llama2 = TextGeneration(generator, prompt=FULL_PROMPT)
    representation_model = {"KeyBERT": keybert, "Llama2": llama2, "MMR": mmr}

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        representation_model=representation_model,
        top_n_words=10,
        verbose=True,
    )

    topics, probs = topic_model.fit_transform(docs)
    print(topic_model.get_topic_info()[1:3])

    # Apply Llama 2 labels and visualize
    llama2_labels = [
        label[0][0].split("\n")[0]
        for label in topic_model.get_topics(full=True)["Llama2"].values()
    ]
    topic_model.set_topic_labels(llama2_labels)
    topic_model.visualize_documents(
        titles,
        reduced_embeddings=reduced_embeddings,
        hide_annotations=True,
        hide_document_hover=False,
        custom_labels=True,
    )


def get_dataset(src, ctry):
    """
    Retrieve and filter documents from MongoDB for topic modeling.

    Parameters
    ----------
    src : list of str
        Source country trigraph codes to include.
    ctry : list of str
        NER target country trigraph codes to filter by.

    Returns
    -------
    tuple
        (corpus, mongo_ids, country_ids, ner_data) arrays.
    """
    client = MongoClient(MONGODB_URI)
    collection = client[MONGO_DB_NAME]["Gov_Corpus"]

    df = pd.DataFrame(
        list(collection.find(
            {}, {"_id", "body", "country", "NER_results", "BERTopic_results"}
        ))
    )
    df = df[df["body"].notna()].reset_index(drop=True)
    df = df[df["country"].isin(src)]
    df = df[df["NER_results"].apply(lambda x: bool(set(x) & set(ctry)))]

    print(f"Filtered results: {len(df)}")

    # Exclude already-processed documents
    try:
        already_done = df[df["BERTopic_results"].notna()]["_id"]
        df = df[~df["_id"].isin(set(already_done))]
    except KeyError:
        print("No existing BERTopic results found — processing all documents.")

    # Flatten body field
    df["body_full"] = df["body"].apply(
        lambda b: " ".join(b) if isinstance(b, list) else b
    )

    return df["body_full"].values, df["_id"], df["country"], df["NER_results"]


def main():
    """Train topic model for Russian documents mentioning Ukraine."""
    ctry = ["UKR"]
    src = ["RUS"]

    corpus, mongo_ids, country_ids, ner_data = get_dataset(src, ctry)
    items = (corpus, mongo_ids, country_ids, ner_data)
    run_topic_model(items)


if __name__ == "__main__":
    main()
