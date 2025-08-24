#!/usr/bin/env python
# metrics.py

# Reference: Hugging Face's Transformers Documentation (/Resources/Token Classification) - https://huggingface.co/docs/transformers/tasks/token_classification
# Reference: Hugging Face LLM Course Chapter 7

import numpy as np
import evaluate

# Load seqeval metric
metric = evaluate.load("seqeval")

def compute_single_task_metrics(p, id2tag):
    """
    Computes evaluation metrics for a single task using seqeval.
    """
    predictions = p.predictions
    labels = p.label_ids

    predictions = np.argmax(predictions, axis=2)

    # Convert label IDs and prediction IDs back to tags, ignoring -100
    true_labels = [
        [id2tag[label] for (p, label) in zip(prediction, labels_per_sent) if label != -100]
        for prediction, labels_per_sent in zip(predictions, labels)
    ]
    true_predictions = [
        [id2tag[p] for (p, label) in zip(prediction, labels_per_sent) if label != -100]
        for prediction, labels_per_sent in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)

    return {
        "eval_precision": results.get("overall_precision"),
        "eval_recall": results.get("overall_recall"),
        "eval_f1": results.get("overall_f1"),
        "eval_accuracy": results.get("overall_accuracy"),
    }


def compute_multitask_metrics(p, id2lid, id2pos):
    """
    Computes evaluation metrics for both LID and POS tasks using seqeval.
    Handles cases where POS labels might be missing in the evaluation data.
    """
    lid_predictions = p.predictions[0]
    pos_predictions = p.predictions[1]
    lid_labels = p.label_ids[0]
    pos_labels = p.label_ids[1]

    metrics = {}

    # Compute LID Metrics
    if lid_labels is not None:
        lid_predicted_label_ids = np.argmax(lid_predictions, axis=2)
        true_lid_labels = [
            [id2lid[label] for (pred, label) in zip(prediction, labels_per_sent) if label != -100]
            for prediction, labels_per_sent in zip(lid_predicted_label_ids, lid_labels)
        ]
        true_lid_predictions = [
            [id2lid[pred] for (pred, label) in zip(prediction, labels_per_sent) if label != -100]
            for prediction, labels_per_sent in zip(lid_predicted_label_ids, lid_labels)
        ]

        if true_lid_labels and true_lid_predictions: # Only compute if there are actual labels
             lid_results = metric.compute(predictions=true_lid_predictions, references=true_lid_labels)
             metrics.update({
                 "eval_lid_precision": lid_results.get("overall_precision"),
                 "eval_lid_recall": lid_results.get("overall_recall"),
                 "eval_lid_f1": lid_results.get("overall_f1"),
                 "eval_lid_accuracy": lid_results.get("overall_accuracy"),
             })
        else:
             print("Warning: No valid LID labels found for metric computation in this batch.")


    # Compute POS Metrics
    if pos_labels is not None and np.any(pos_labels != -100):
        pos_predicted_label_ids = np.argmax(pos_predictions, axis=2)
        true_pos_labels = [
            [id2pos[label] for (pred, label) in zip(prediction, labels_per_sent) if label != -100]
            for prediction, labels_per_sent in zip(pos_predicted_label_ids, pos_labels)
        ]
        true_pos_predictions = [
            [id2pos[pred] for (pred, label) in zip(prediction, labels_per_sent) if label != -100]
            for prediction, labels_per_sent in zip(pos_predicted_label_ids, pos_labels)
        ]

        if true_pos_labels and true_pos_predictions:
             pos_results = metric.compute(predictions=true_pos_predictions, references=true_pos_labels)
             metrics.update({
                 "eval_pos_precision": pos_results.get("overall_precision"),
                 "eval_pos_recall": pos_results.get("overall_recall"),
                 "eval_pos_f1": pos_results.get("overall_f1"),
                 "eval_pos_accuracy": pos_results.get("overall_accuracy"),
             })
        else:
             print("Warning: No valid POS labels found for metric computation in this batch.")


    # Compute combined F1 if both are available
    if "eval_lid_f1" in metrics and "eval_pos_f1" in metrics:
        metrics["eval_combined_f1"] = (metrics["eval_lid_f1"] + metrics["eval_pos_f1"]) / 2.0 # Average of LID and POS F1
    elif "eval_lid_f1" in metrics:
         metrics["eval_combined_f1"] = metrics["eval_lid_f1"] # If only LID is available, use LID F1
    elif "eval_pos_f1" in metrics:
         metrics["eval_combined_f1"] = metrics["eval_pos_f1"] # If only POS is available, use POS F1

    return metrics