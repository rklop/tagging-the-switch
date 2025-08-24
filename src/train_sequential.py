#!/usr/bin/env python
# train_sequential.py

# Reference: Hugging Face's Transformers Documentation (/Resources/Token Classification)
# Reference: Hugging Face LLM Course Chapter 3 and 7
# https://huggingface.co/docs/transformers/tasks/token_classification

import os
import torch
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
from utils.data import load_and_preprocess_data, tokenizer 
from utils.metrics import compute_single_task_metrics       
from config.config import ( 
    MODEL_NAME, BASE_DIR, SAVES_DIR,
    SINGLE_TASK_POS_SAVE_PATH, SINGLE_TASK_LID_SAVE_PATH,
    SEQUENTIAL_POS_LID_SAVE_PATH, SEQUENTIAL_LID_POS_SAVE_PATH,
    LEARNING_RATE, NUM_TRAIN_EPOCHS, PER_DEVICE_TRAIN_BATCH_SIZE, 
    PER_DEVICE_EVAL_BATCH_SIZE, WEIGHT_DECAY, METRIC_FOR_BEST_MODEL_SINGLE_TASK, MODEL_SAVE_DIR
)

def train_sequential_model(first_task, second_task):
    """
    Trans a model sequentially on two task (first_task and then second_task).
    """
    if first_task not in ['pos', 'lid'] or second_task not in ['pos', 'lid'] or first_task == second_task:
        raise ValueError("first_task and second_task must be 'pos' or 'lid' and different.")

    print(f"Starting Sequential Training: {first_task.upper()} then {second_task.upper()}\n\n")

    # Train on First Task
    print(f"\nStarting Phase 1: Training on {first_task.upper()}...")

    # Load and preprocess data for the first task
    train_dataset_first, dev_dataset_first, label2id_first, id2tag_first = load_and_preprocess_data(first_task)
    num_labels_first = len(label2id_first)

    # Load the base model for the first task
    model_first = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels_first,
        id2label=id2tag_first,
        label2id=label2id_first
    )

    # Define TrainingArguments for the first phase
    output_dir_first = os.path.join(SAVES_DIR, f'sequential_phase1_{first_task}')

    args_first = TrainingArguments(
        output_dir=output_dir_first,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        logging_strategy="steps",
        logging_steps=10,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        num_train_epochs=NUM_TRAIN_EPOCHS, 
        weight_decay=WEIGHT_DECAY,
        remove_unused_columns=False,
        report_to="none",
        fp16=torch.cuda.is_available(),
        load_best_model_at_end=True,
        metric_for_best_model=METRIC_FOR_BEST_MODEL_SINGLE_TASK, 
        greater_is_better=True
    )

    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer)

    # Create Trainer for the first phase
    trainer_first = Trainer(
        model=model_first,
        args=args_first,
        train_dataset=train_dataset_first,
        eval_dataset=dev_dataset_first,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda p: compute_single_task_metrics(p, id2tag_first) # Metrics for the first task
    )

    # Train the model on the first task
    trainer_first.train()

    # Save the model after the first phase
    phase1_save_path = os.path.join(MODEL_SAVE_DIR, f'sequential_phase1_{first_task}-final')
    os.makedirs(os.path.dirname(phase1_save_path), exist_ok=True)
    model_first.save_pretrained(phase1_save_path)
    tokenizer.save_pretrained(phase1_save_path)

    print(f"Phase 1 ({first_task.upper()}) training complete. Model saved to {phase1_save_path}")

    # Fine-tune on Second Task
    print(f"\nStarting Phase 2: Fine-tuning on {second_task.upper()}...")

    # Load and preprocess data for the second task
    train_dataset_second, dev_dataset_second, label2id_second, id2tag_second = load_and_preprocess_data(second_task)
    num_labels_second = len(label2id_second)

    # Load the model saved after the first phase for fine-tuning
    # Make sure ignore_mismatched_sizes is set to True; to handle the different number of output labels from each task's model
    model_second = AutoModelForTokenClassification.from_pretrained(
        phase1_save_path, 
        num_labels=num_labels_second,
        id2label=id2tag_second, 
        label2id=label2id_second, 
        ignore_mismatched_sizes=True # Ignore mismatch in classifie head
    )

    # Define TrainingArguments for the second phase
    output_dir_second = os.path.join(SAVES_DIR, f'sequential_phase2_{second_task}')
    # Determine final save path based on sequential order
    final_save_path = SEQUENTIAL_POS_LID_SAVE_PATH if first_task == 'pos' else SEQUENTIAL_LID_POS_SAVE_PATH

    args_second = TrainingArguments(
        output_dir=output_dir_second,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        logging_strategy="steps",
        logging_steps=10,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        num_train_epochs=NUM_TRAIN_EPOCHS, 
        weight_decay=WEIGHT_DECAY,
        remove_unused_columns=False,
        report_to="none",
        fp16=torch.cuda.is_available(),
        load_best_model_at_end=True,
        metric_for_best_model=METRIC_FOR_BEST_MODEL_SINGLE_TASK, 
        greater_is_better=True
    )

    # Create Trainer for the second phase
    trainer_second = Trainer(
        model=model_second,
        args=args_second,
        train_dataset=train_dataset_second,
        eval_dataset=dev_dataset_second,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda p: compute_single_task_metrics(p, id2tag_second) # Metrics for the second task
    )

    # Train the model on the second task
    trainer_second.train()

    # Evaluate on the dev set after training
    print(f"Evaluating sequential ({first_task.upper()} then {second_task.upper()}) model on development set...")
    eval_results = trainer_second.evaluate(dev_dataset_second)
    print("\n\n\n ##########################################")
    print(f"Sequential ({first_task.upper()} then {second_task.upper()}) Evaluation Results:", eval_results)
    print("##########################################\n\n\n")

    # Save the final sequentia fine-tuned model
    print(f"Saving sequentially fine-tuned model to {final_save_path}")
    os.makedirs(os.path.dirname(final_save_path), exist_ok=True)
    model_second.save_pretrained(final_save_path)
    tokenizer.save_pretrained(final_save_path) # Save tokenizer too

    print(f"--- Sequential Training ({first_task.upper()} then {second_task.upper()}) Complete ---")


if __name__ == '__main__':
    # Train POS then LID
    train_sequential_model('pos', 'lid')

    # Train LID then POS
    train_sequential_model('lid', 'pos')