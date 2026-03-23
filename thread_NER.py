"""
thread_NER.py
=============
Named Entity Recognition (NER) pipeline for the Russia-Colombia corpus.

Runs a multilingual NER model (Babelscape/wikineural-multilingual-ner) over
document texts from MongoDB, extracts location, person, organization, and
miscellaneous entities, converts location entities to ISO 3166-1 alpha-3
trigraph codes using country_converter, and writes results back to MongoDB.

Uses multiprocessing to parallelize inference across CPU cores.
"""

import multiprocessing
import time

import country_converter as coco
import pandas as pd
from bson.objectid import ObjectId
from pymongo import MongoClient
from torch.multiprocessing import Pool, set_start_method
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

from config import (
    MONGODB_URI,
    MONGO_COLLECTION_NAME,
    MONGO_DB_NAME,
    NER_MODEL_NAME,
)

set_start_method("spawn", force=True)


def get_collection():
    """Return the MongoDB collection handle."""
    client = MongoClient(MONGODB_URI)
    return client[MONGO_DB_NAME][MONGO_COLLECTION_NAME]


def run_NER(items):
    """
    Run NER on a single document and update MongoDB with results.

    Parameters
    ----------
    items : tuple of (str, ObjectId, str)
        (document_text, mongo_id, source_identifier).
    """
    text, mongo_id, source_id = items

    tokenizer = AutoTokenizer.from_pretrained(NER_MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(NER_MODEL_NAME)
    nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="average")

    ner_results = nlp(text)
    mongo_update(ner_results, mongo_id, source_id)


def get_dataset(collection):
    """
    Retrieve documents from MongoDB that still need NER processing.

    Parameters
    ----------
    collection : pymongo.collection.Collection
        Source MongoDB collection.

    Returns
    -------
    tuple of (ndarray, Series, Series)
        Document texts, MongoDB ObjectIds, and source identifiers.
    """
    df = pd.DataFrame(
        list(collection.find({}, {"_id", "body", "website", "NER_Country"}))
    )
    df = df[df["body"].notna()].reset_index(drop=True)
    print(f"Full dataset: {len(df)}")

    # Exclude already-processed documents
    try:
        already_done = df[df["NER_Country"].notna()]["_id"]
        df = df[~df["_id"].isin(set(already_done))]
    except KeyError:
        print("No existing NER results found — processing all documents.")

    print(f"Updates needed: {len(df)}")

    # Flatten body field (may be list of paragraphs or a single string)
    df["body_full"] = df["body"].apply(
        lambda b: " ".join(b) if isinstance(b, list) else b
    )

    return df["body_full"].values, df["_id"], df["website"]


def mongo_update(ner_results, mongo_id, source_id):
    """
    Parse NER results by entity type and update MongoDB.

    Parameters
    ----------
    ner_results : list of dict
        Raw NER pipeline output.
    mongo_id : ObjectId
        MongoDB document ID.
    source_id : str
        Source identifier (used to exclude self-references in trigraph conversion).
    """
    collection = get_collection()
    df = pd.DataFrame(ner_results)

    ner_loc, ner_per, ner_org, ner_mis = [], [], [], []

    for entity_type, target_list in [
        ("LOC", ner_loc), ("PER", ner_per), ("ORG", ner_org), ("MIS", ner_mis)
    ]:
        try:
            target_list.extend(
                df[df["entity_group"].str.contains(entity_type)]["word"].tolist()
            )
        except (KeyError, AttributeError):
            pass

    trigraph_list = trigraph_convert(ner_loc, source_id) if ner_loc else []

    update_doc = {
        "$set": {
            "NER_Country": trigraph_list,
            "NER_Location": ner_loc,
            "NER_Person": ner_per,
            "NER_Organization": ner_org,
            "NER_Misc": ner_mis,
        }
    }

    collection.find_one_and_update({"_id": ObjectId(mongo_id)}, update_doc)


def trigraph_convert(ner_locations, source_id):
    """
    Convert location entity strings to ISO 3166-1 alpha-3 codes.

    Parameters
    ----------
    ner_locations : list of str
        Location entity strings extracted by NER.
    source_id : str
        Source identifier to exclude from results (avoids trivial self-matches).

    Returns
    -------
    list of str
        Deduplicated ISO trigraph codes.
    """
    cc = coco.CountryConverter()
    try:
        trigraphs = list(set(
            t for t in [cc.convert(loc, to="iso3") for loc in ner_locations]
            if t != "not found" and t != source_id
        ))
    except Exception:
        trigraphs = ["error"]
    return trigraphs


def main():
    """Run NER across all unprocessed documents using multiprocessing."""
    start = time.time()
    print("Script activated")

    collection = get_collection()
    corpus, mongo_ids, source_ids = get_dataset(collection)
    items = list(zip(corpus, mongo_ids, source_ids))

    cpu_count = multiprocessing.cpu_count()
    print(f"CPU count: {cpu_count}")

    pool = Pool(processes=cpu_count)
    pool.map(run_NER, items)
    pool.close()
    pool.join()

    print(f"Total time: {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
