#!/usr/bin/env python
# train_multitask_unweighted.py

import os
import torch
from transformers import TrainingArguments, Trainer, DataCollatorForTokenClassification, AutoModel
from utils.data import load_and_preprocess_data, tokenizer 
from utils.multitaskmodel import MultitaskTokenClassification 
from utils.metrics import compute_multitask_metrics         
from config.config import ( 
    MODEL_NAME, BASE_DIR, SAVES_DIR, MULTITASK_UNWEIGHTED_SAVE_PATH,
    LEARNING_RATE, NUM_TRAIN_EPOCHS, PER_DEVICE_TRAIN_BATCH_SIZE,
    PER_DEVICE_EVAL_BATCH_SIZE, WEIGHT_DECAY, METRIC_FOR_BEST_MODEL_MULTITASK
)

def train_multitask_unweighted_model():
    """
    Trains a joint multitask model with (un)weighted loss for LID and POS.
    """

    # Load and preprocess multitask data
    train_dataset, dev_dataset, lid2id, id2lid, pos2id, id2pos = load_and_preprocess_data('multitask')

    # Define the base model configuration and add num_labels for each task
    config = AutoModel.from_pretrained(MODEL_NAME).config
    config.num_labels_lid = len(lid2id)
    config.num_labels_pos = len(pos2id)

    # Instantiate the custom multitask model
    model = MultitaskTokenClassification(config=config)

    # Define TrainingArguments
    output_dir = os.path.join(SAVES_DIR, 'multitask_unweighted')
    model_save_path = MULTITASK_UNWEIGHTED_SAVE_PATH

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
        fp16=torch.cuda.is_available(),
        load_best_model_at_end=True,
        metric_for_best_model=METRIC_FOR_BEST_MODEL_MULTITASK,
        greater_is_better=True
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    # Create Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda p: compute_multitask_metrics(p, id2lid, id2pos)
    )

    # Train the model
    print("Training joint multitask (un)weighted model...")
    trainer.train()

    # Evaluate on the development set after training
    print("Evaluating joint multitask (un)weighted model on development set...")
    eval_results = trainer.evaluate(dev_dataset)
    print("\n\n\n ##########################################")
    print("Joint Multitask (Un)weighted Evaluation Results:", eval_results)
    print("##########################################\n\n\n")
    
    # Save the final best model
    print(f"Saving fine-tuned joint multitask (un)weighted model to {model_save_path}")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path) # Save tokenizer

if __name__ == '__main__':
    train_multitask_unweighted_model()