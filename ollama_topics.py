"""
ollama_topics.py
================
LLM-based topic label generation using Ollama (local inference).

Reads BERTopic output (topic IDs, representative documents, KeyBERT keywords)
from CSV files, prompts a local Ollama model to generate human-readable
topic labels, and saves the labeled results.

This is a post-processing step that runs after Topic_Cluster.py.
"""

import ast
import os
import re

import ollama
import pandas as pd

from config import KEYBERT_LABELS_DIR, OLLAMA_MODEL


def topic_postprocessing(df_cons):
    """
    Generate LLM topic labels for each topic cluster.

    For each unique topic, sends representative documents and KeyBERT
    keywords to the Ollama model and requests three short label candidates.

    Parameters
    ----------
    df_cons : pd.DataFrame
        DataFrame with columns: Topic, Representative_Docs, KeyBERT.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with an added 'Ollama_Topics' column containing
        lists of label candidates per topic.
    """
    topic_ids = df_cons["Topic"]
    rep_docs = df_cons["Representative_Docs"]
    keybert_keywords = df_cons["KeyBERT"]

    rep_docs_parsed = [ast.literal_eval(x) for x in rep_docs]
    df_topics = pd.DataFrame(
        {"Topic": topic_ids, "Representative_Docs": rep_docs_parsed, "KeyBERT": keybert_keywords}
    )
    df_topics.drop_duplicates(inplace=True, subset=["Topic"])

    system_prompt = "You are a helpful, respectful and honest assistant for labeling topics."

    example_prompt = (
        "I have a topic that contains the following documents:\n"
        "- Traditional diets in most cultures were primarily plant-based with a "
        "little meat on top, but with the rise of industrial style meat production "
        "and factory farming, meat has become a staple food.\n"
        "- Meat, but especially beef, is the word food in terms of emissions.\n"
        "- Eating meat doesn't make you a bad person, not eating meat doesn't make "
        "you a good one.\n\n"
        "The topic is described by the following keywords: 'meat, beef, eat, eating, "
        "emissions, steak, food, health, processed, chicken'.\n\n"
        "Based on the information about the topic above, please create a short label "
        "of this topic. Make sure you to only return the label and nothing more.\n"
        "[/INST] Environmental impacts of eating meat"
    )

    variants = []
    for keywords, docs in zip(df_topics["KeyBERT"], df_topics["Representative_Docs"]):
        document = " ".join(docs)
        user_prompt = (
            f"I have a topic that contains the following documents: {document}\n"
            f"The topic is described by the following keywords: {keywords}\n"
            "Based on the information about the topic above, please create 3 short "
            "labels of this topic. Make sure you to only return the label and nothing more."
        )

        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "assistant", "content": example_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        label_text = response["message"]["content"]
        print(label_text)
        variants.append(label_text)

    # Parse numbered label lists into flat lists of strings
    variants_parsed = [
        x.strip().split(",")
        for x in [re.sub(r"[1.2.3.]", "", x) for x in [x.replace("\n", ",") for x in variants]]
    ]

    topic_dict = dict(zip(df_topics["Topic"], variants_parsed))
    df_cons["Ollama_Topics"] = df_cons["Topic"].map(topic_dict)

    return df_cons


def get_dataset(filepath):
    """
    Load a KeyBERT labels CSV file.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
    """
    return pd.read_csv(filepath)


def main():
    """Process all KeyBERT label files in the labels directory."""
    dir_list = os.listdir(KEYBERT_LABELS_DIR)
    print(f"Files found: {dir_list}")

    for filename in dir_list:
        if "Labels" in filename:
            print(f"Processing: {filename}")
            filepath = os.path.join(KEYBERT_LABELS_DIR, filename)
            dataset = get_dataset(filepath)
            label_df = topic_postprocessing(dataset)
            save_path = os.path.join(KEYBERT_LABELS_DIR, "REVIEW_" + filename)
            label_df.to_csv(save_path, index=False)
            print(f"Saved to: {save_path}")


if __name__ == "__main__":
    main()
