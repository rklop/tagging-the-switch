#!/usr/bin/env python
# train_single_task.py

# Reference: Hugging Face's Transformers Documentation (/Resources/Token Classification)
# Reference: Hugging Face LLM Course Chapter 3 and 7
# https://huggingface.co/docs/transformers/tasks/token_classification

import os
import torch
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
from utils.data import load_and_preprocess_data, tokenizer 
from utils.metrics import compute_single_task_metrics       
from config.config import (
    MODEL_NAME, BASE_DIR, SAVES_DIR, SINGLE_TASK_POS_SAVE_PATH, SINGLE_TASK_LID_SAVE_PATH,
    LEARNING_RATE, NUM_TRAIN_EPOCHS, PER_DEVICE_TRAIN_BATCH_SIZE,
    PER_DEVICE_EVAL_BATCH_SIZE, WEIGHT_DECAY, METRIC_FOR_BEST_MODEL_SINGLE_TASK
)

def train_single_task_model(task_type):
    """
    Trains a single-task model for either POS or LID.
    """

    print(f"Starting Single-task {task_type.upper()} Training\n\n")

    # Load and preprocess data
    train_dataset, dev_dataset, label2id, id2tag = load_and_preprocess_data(task_type)
    num_labels = len(label2id)

    # Load the base model with a token classification head
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label=id2tag,
        label2id=label2id
    )

    # Define TrainingArguments
    output_dir = os.path.join(SAVES_DIR, f'single_task_{task_type}')
    model_save_path = SINGLE_TASK_POS_SAVE_PATH if task_type == 'pos' else SINGLE_TASK_LID_SAVE_PATH

    args = TrainingArguments(
        output_dir=output_dir,
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
        fp16=torch.cuda.is_available(), # Use fp16 if GPU is available for faster training and lower memory usage
        load_best_model_at_end=True,
        metric_for_best_model=METRIC_FOR_BEST_MODEL_SINGLE_TASK,
        greater_is_better=True
    )

    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer)

    # Create Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda p: compute_single_task_metrics(p, id2tag) # Pass id2tag to metrics function
    )

    # Train the model
    print(f"Training single-task {task_type.upper()} model \n\n")
    trainer.train()

    # Evaluate on the development set after training (skipped if eval_dataset is None)
    print(f"Evaluating single-task {task_type.upper()} model on development set...")
    eval_results = trainer.evaluate(dev_dataset)
    print("\n\n\n ##########################################")
    print(f"Single-task {task_type.upper()} Evaluation Results:", eval_results)
    print("##########################################\n\n\n")

    # Save the final best model
    print(f"Saving fine-tuned single-task {task_type.upper()} model to {model_save_path}")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path) # Save the tokenizer

    print(f"Single-task {task_type.upper()} Training Complete")

if __name__ == '__main__':
    # Train POS model
    train_single_task_model('pos')
    # Train LID model
    train_single_task_model('lid')