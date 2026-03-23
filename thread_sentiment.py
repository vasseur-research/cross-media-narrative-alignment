"""
thread_sentiment.py
===================
Sentiment analysis pipeline for the Russia-Colombia narrative corpus.

Applies a multilingual BERT sentiment model (nlptown/bert-base-multilingual-
uncased-sentiment) to documents stored in MongoDB. Long documents are split
into 512-token chunks, scored independently, and averaged to produce a
single 1–5 sentiment rating per document. Results are written back to MongoDB.

Uses multiprocessing to parallelize inference across CPU cores.
"""

import multiprocessing
import time

import pandas as pd
import torch
from bson.objectid import ObjectId
from pymongo import MongoClient
from torch.multiprocessing import Pool, set_start_method
from transformers import BertForSequenceClassification, BertTokenizer

from config import (
    MONGODB_URI,
    MONGO_COLLECTION_NAME,
    MONGO_DB_NAME,
    SENTIMENT_CHUNK_SIZE,
    SENTIMENT_MODEL_NAME,
)

set_start_method("spawn", force=True)


def get_collection():
    """Return the MongoDB collection handle."""
    client = MongoClient(MONGODB_URI)
    return client[MONGO_DB_NAME][MONGO_COLLECTION_NAME]


def run_sentiment(items):
    """
    Score sentiment for a single document using chunked BERT inference.

    Long documents are split into overlapping chunks of SENTIMENT_CHUNK_SIZE
    tokens. Each chunk is padded/truncated, scored independently, and the
    softmax probabilities are averaged across chunks. The final sentiment
    is the argmax of the averaged distribution (1-indexed, 1–5 scale).

    Parameters
    ----------
    items : tuple of (str, ObjectId)
        (document_text, mongo_id).

    Returns
    -------
    int
        Sentiment score on a 1–5 scale.
    """
    text, doc_id = items

    model = BertForSequenceClassification.from_pretrained(SENTIMENT_MODEL_NAME)
    tokenizer = BertTokenizer.from_pretrained(SENTIMENT_MODEL_NAME)

    tokens = tokenizer.encode_plus(text, add_special_tokens=False, return_tensors="pt")
    chunksize = SENTIMENT_CHUNK_SIZE

    # Split into chunks of (chunksize - 2) to leave room for CLS/SEP tokens
    input_id_chunks = list(tokens["input_ids"][0].split(chunksize - 2))
    mask_chunks = list(tokens["attention_mask"][0].split(chunksize - 2))

    for i in range(len(input_id_chunks)):
        # Add CLS (101) and SEP (102) tokens
        input_id_chunks[i] = torch.cat([
            torch.tensor([101]), input_id_chunks[i], torch.tensor([102])
        ])
        mask_chunks[i] = torch.cat([
            torch.tensor([1]), mask_chunks[i], torch.tensor([1])
        ])

        # Pad to chunksize if needed
        pad_len = chunksize - input_id_chunks[i].shape[0]
        if pad_len > 0:
            input_id_chunks[i] = torch.cat([
                input_id_chunks[i], torch.Tensor([0] * pad_len)
            ])
            mask_chunks[i] = torch.cat([
                mask_chunks[i], torch.Tensor([0] * pad_len)
            ])

    input_ids = torch.stack(input_id_chunks)
    attention_mask = torch.stack(mask_chunks)

    input_dict = {
        "input_ids": input_ids.long(),
        "attention_mask": attention_mask.int(),
    }

    outputs = model(**input_dict)
    probs = torch.nn.functional.softmax(outputs[0], dim=-1)
    probs = probs.mean(dim=0)

    # Convert 0-indexed argmax to 1-indexed sentiment score
    sentiment = torch.argmax(probs).item() + 1

    collection = get_collection()
    collection.find_one_and_update(
        {"_id": ObjectId(doc_id)},
        {"$set": {"Sentiment": sentiment}},
    )

    return sentiment


def get_dataset(collection):
    """
    Retrieve documents from MongoDB that still need sentiment scoring.

    Parameters
    ----------
    collection : pymongo.collection.Collection
        Source MongoDB collection.

    Returns
    -------
    tuple of (list of str, Series)
        Document texts and their MongoDB ObjectIds.
    """
    df = pd.DataFrame(
        list(collection.find({}, {"_id", "body", "Sentiment"}))
    )
    df = df[df["body"].notna()].reset_index(drop=True)

    # Exclude already-scored documents
    try:
        already_done = df[df["Sentiment"].notna()]["_id"]
        df = df[~df["_id"].isin(set(already_done))]
    except KeyError:
        print("No existing sentiment scores found — processing all documents.")

    df["bodyFull"] = df["body"]
    return df["bodyFull"].tolist(), df["_id"]


def main():
    """Run sentiment analysis across all unprocessed documents."""
    start = time.time()
    print("Script activated")

    collection = get_collection()
    corpus, mongo_ids = get_dataset(collection)
    items = list(zip(corpus, mongo_ids))

    db_time = time.time()
    print(f"Dataset load time: {db_time - start:.1f}s")

    cpu_count = multiprocessing.cpu_count()
    print(f"CPU count: {cpu_count}")

    pool = Pool(processes=cpu_count)
    predictions = pool.map(run_sentiment, items)
    pool.close()
    pool.join()

    print(f"Inference time: {time.time() - db_time:.1f}s")
    print(f"Total time: {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
