#!/usr/bin/env python
# data_utils.py

# Reference: Hugging Face's Transformers Documentation (/Resources/Token Classification) - https://huggingface.co/docs/transformers/tasks/token_classification
# Reference: Hugging Face LLM Course Chapter 7 


import os
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from config.config import (MODEL_NAME, MAX_SEQUENCE_LENGTH, 
    POS_DEV_FILE, POS_TRAIN_FILE, LID_DEV_FILE, LID_TRAIN_FILE)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def load_conll_data(filepath, task_type='pos'):
    """
    Loads data from a .conll file based on task type passed in params.
    Assumes each line has a token and annotations separated by tabs.
    """
    sentences = []
    tokens = []
    tags = [] # For single task (POS or LID)
    lid_tags = [] # For multitask
    pos_tags = [] # For multitask

    print(f"Loading {task_type} data from {filepath}")
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    if tokens:
                        sentence_data = {"tokens": tokens}
                        if task_type == 'pos':
                            sentence_data["pos_tags"] = tags
                        elif task_type == 'lid':
                            sentence_data["lid_tags"] = tags
                        elif task_type == 'multitask':
                             sentence_data["lid_tags"] = lid_tags
                             sentence_data["pos_tags"] = pos_tags
                        else:
                            raise ValueError(f"Unknown task type: {task_type}")

                        # Append sentence data and reset lists befre processing next sentencein file
                        sentences.append(sentence_data)
                        tokens = []
                        tags = []
                        lid_tags = []
                        pos_tags = []
                    continue

                parts = line.split()
                if task_type == 'pos' and len(parts) >= 3:
                    token, _, pos = parts[0], parts[1], parts[2] # Assuming token\tlang\tpos
                    tokens.append(token)
                    tags.append(pos)
                elif task_type == 'lid' and len(parts) >= 2:
                     token, lid = parts[0], parts[1] # Assuming token\tlang
                     tokens.append(token)
                     tags.append(lid)
                elif task_type == 'multitask' and len(parts) >= 3:
                     token, lid, pos = parts[0], parts[1], parts[2] # Assuming token\tlang\tpos
                     tokens.append(token)
                     lid_tags.append(lid)
                     pos_tags.append(pos)
                # else:
                    # print(f"Warning: Skipping malformed line in {filepath}: {line}")


        # Add the last sentence if the file doesn't end with a blank line
        if tokens:
             sentence_data = {"tokens": tokens}
             if task_type == 'pos':
                 sentence_data["pos_tags"] = tags
             elif task_type == 'lid':
                 sentence_data["lid_tags"] = tags
             elif task_type == 'multitask':
                  sentence_data["lid_tags"] = lid_tags
                  sentence_data["pos_tags"] = pos_tags
             sentences.append(sentence_data)

        print(f"Loaded {len(sentences)} sentencs for {task_type}.")
        return sentences
    except FileNotFoundError:
        print(f"Error: Data file not found at {filepath}.")
        return [] # Return empty list if file not found


def build_vocab(data_splits, tag_key):
    """
    Builds a vocabulary of unique tags for a specific task from data splits.
    """
    all_tags = []
    for data_split in data_splits:
        for sentence in data_split:
            all_tags.extend(sentence.get(tag_key, [])) # Use .get for safety

    unique_tags = sorted(list(set(all_tags)))
    tag2id = {tag: i for i, tag in enumerate(unique_tags)}
    id2tag = {i: tag for tag, i in tag2id.items()}

    print(f"Built {tag_key} vocab with {len(tag2id)} tags: {unique_tags}")

    return tag2id, id2tag


def align_labels_with_tokens(labels, word_ids):
    """
    Aligns labels with tokenized inputs, assigning -100 to subword tokens
    that do not correspond to the start of a word.
    """
    aligned_labels = []
    prev_word_idx = None
    for word_idx in word_ids:
        if word_idx is None:
            # Special tokens (like [CLS], [SEP], [PAD]) get -100
            aligned_labels.append(-100)
        elif word_idx != prev_word_idx:
            # Assign label to the first token of a word
            aligned_labels.append(labels[word_idx])
        else:
            # Subword tokens subsequent to the first token get -100
            aligned_labels.append(-100)
        prev_word_idx = word_idx
    return aligned_labels


def tokenize_and_align_single_task(batch, label2id, tag_key):
    """
    Tokenizes a batch of sentences and aligns single-task labels.
    """
    tokens = batch["tokens"]
    tags = batch[tag_key]
    labels = [[label2id[tag] for tag in sentence_tags] for sentence_tags in tags]

    tokenized_inputs = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
        max_length=MAX_SEQUENCE_LENGTH,
    )

    all_labels = []
    for i, sentence_labels in enumerate(labels):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        aligned_labels = align_labels_with_tokens(sentence_labels, word_ids)
        all_labels.append(aligned_labels)

    tokenized_inputs["labels"] = all_labels

    return tokenized_inputs


def tokenize_and_align_multitask(batch, lid2id, pos2id):
    """
    Tokenizes a batch of sentnces and aligns both LID and POS labels for multitask.
    """
    tokens = batch["tokens"]
    lid_tags = batch["lid_tags"]
    pos_tags = batch["pos_tags"]

    lid_labels = [[lid2id[tag] for tag in sentence_lid_tags] for sentence_lid_tags in lid_tags]
    pos_labels = [[pos2id[tag] for tag in sentence_pos_tags] for sentence_pos_tags in pos_tags]

    tokenized_inputs = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
        max_length=MAX_SEQUENCE_LENGTH,
    )

    all_aligned_lid_labels = []
    all_aligned_pos_labels = []

    for i, (sentence_lid_labels, sentence_pos_labels) in enumerate(zip(lid_labels, pos_labels)):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        aligned_lid = align_labels_with_tokens(sentence_lid_labels, word_ids)
        aligned_pos = align_labels_with_tokens(sentence_pos_labels, word_ids)
        all_aligned_lid_labels.append(aligned_lid)
        all_aligned_pos_labels.append(aligned_pos)

    tokenized_inputs["labels_lid"] = all_aligned_lid_labels
    tokenized_inputs["labels_pos"] = all_aligned_pos_labels

    return tokenized_inputs

def load_and_preprocess_data(task_type):
    """
    Loads and preprocesses data for a given task type.
    """
    if task_type == 'pos':
        train_file = POS_TRAIN_FILE
        dev_file = POS_DEV_FILE

        train_data = load_conll_data(train_file, task_type='pos')
        dev_data = load_conll_data(dev_file, task_type='pos')

        pos2id, id2pos = build_vocab([train_data, dev_data], "pos_tags")

        train_dataset = Dataset.from_list(train_data).map(
            lambda batch: tokenize_and_align_single_task(batch, pos2id, 'pos_tags'),
            batched=True, remove_columns=["tokens", "pos_tags"]
        )
        dev_dataset = Dataset.from_list(dev_data).map(
            lambda batch: tokenize_and_align_single_task(batch, pos2id, 'pos_tags'),
            batched=True, remove_columns=["tokens", "pos_tags"]
        )

        return train_dataset, dev_dataset, pos2id, id2pos

    elif task_type == 'lid':
        train_file = LID_TRAIN_FILE
        dev_file = LID_DEV_FILE

        train_data = load_conll_data(train_file, task_type='lid')
        dev_data = load_conll_data(dev_file, task_type='lid')

        lid2id, id2lid = build_vocab([train_data, dev_data], "lid_tags")

        train_dataset = Dataset.from_list(train_data).map(
            lambda batch: tokenize_and_align_single_task(batch, lid2id, 'lid_tags'),
            batched=True, remove_columns=["tokens", "lid_tags"]
        )
        dev_dataset = Dataset.from_list(dev_data).map(
            lambda batch: tokenize_and_align_single_task(batch, lid2id, 'lid_tags'),
            batched=True, remove_columns=["tokens", "lid_tags"]
        )

        return train_dataset, dev_dataset, lid2id, id2lid

    elif task_type == 'multitask':
        # For multitask training, we assume train/dev have both LID and POS
        train_file = POS_TRAIN_FILE 
        dev_file = POS_DEV_FILE 

        train_data = load_conll_data(train_file, task_type='multitask')
        dev_data = load_conll_data(dev_file, task_type='multitask')

        # Build vocabs for both tasks from train and dev data
        lid2id, id2lid = build_vocab([train_data, dev_data], "lid_tags")
        pos2id, id2pos = build_vocab([train_data, dev_data], "pos_tags")


        train_dataset = Dataset.from_list(train_data).map(
            lambda batch: tokenize_and_align_multitask(batch, lid2id, pos2id),
            batched=True, remove_columns=["tokens", "lid_tags", "pos_tags"]
        )
        dev_dataset = Dataset.from_list(dev_data).map(
            lambda batch: tokenize_and_align_multitask(batch, lid2id, pos2id),
            batched=True, remove_columns=["tokens", "lid_tags", "pos_tags"]
        )

        return train_dataset, dev_dataset, lid2id, id2lid, pos2id, id2pos

    else:
        raise ValueError(f"Unknown task type: {task_type}")