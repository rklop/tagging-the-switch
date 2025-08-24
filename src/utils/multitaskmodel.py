#!/usr/bin/env python
# models.py

# Reference: Model and code structure  (https://github.com/huggingface/transformers/blob/2932f318a20d9e54cc7aea052e040164d85de7d6/src/transformers/models/bert/modeling_bert.py#L1179)
# Reference: More model (XLM-R, mBERT, etc.) definitions/customization (https://github.com/huggingface/transformers/tree/2932f318a20d9e54cc7aea052e040164d85de7d6/src/transformers/models)

import torch
from torch import nn
from transformers import AutoModel, PreTrainedModel
from config.config import MODEL_CONFIG, LID_WEIGHT, POS_WEIGHT, UNCERTAINTY_WEIGHTING

from transformers.modeling_outputs import TokenClassifierOutput
from typing import Optional, Tuple, Union, Dict

class MultitaskTokenClassification(PreTrainedModel):
    """
    Custom modle with separate classification heads for LID and POS tagging.
    """
    # Config class based on specified model name (confirm in config/config.py)
    config_class = MODEL_CONFIG 

    def __init__(self, config):
        super().__init__(config)
        # Ensure config param passed fro train_multitask has num_labels for both tasks
        self.num_labels_lid = config.num_labels_lid
        self.num_labels_pos = config.num_labels_pos

        # Load the base model encoder
        self.model = AutoModel.from_config(config)

        classifier_dropout = getattr(config, 'classifier_dropout', None)
        # If neither dropout is specified, use a default value
        # If missing an error is raised when using while using RemBERT
        if classifier_dropout is None:
            classifier_dropout = getattr(config, 'hidden_dropout_prob', 0.1) # 0.1 default fallback dropout value

        self.dropout = nn.Dropout(classifier_dropout)

        # Separate classification heads for each task
        self.lid_classifier = nn.Linear(config.hidden_size, self.num_labels_lid)
        self.pos_classifier = nn.Linear(config.hidden_size, self.num_labels_pos)

        # Parameters for uncertainty-based weighting
        # Initialize to zeros for equal initial weighting
        self.log_var_pos = nn.Parameter(torch.zeros(1))
        self.log_var_lid = nn.Parameter(torch.zeros(1))

        # Initialize weights
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        # Labels for multitask training
        labels_lid: Optional[torch.Tensor] = None,
        labels_pos: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs # for any addt'l arguments
    ) -> Union[Tuple[torch.Tensor], Dict[str, torch.Tensor]]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Pass inputs through the shared model encoder
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            **kwargs
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)

        # Get logits for each task
        lid_logits = self.lid_classifier(sequence_output)
        pos_logits = self.pos_classifier(sequence_output)

        loss_lid = None
        loss_pos = None
        loss = None

        # Calculate losses if labels are provided for both tasks
        if labels_lid is not None and labels_pos is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100) # Ignore index for padded tokens

            # Calculate loss for each task
            loss_lid = loss_fct(lid_logits.view(-1, self.num_labels_lid), labels_lid.view(-1))
            loss_pos = loss_fct(pos_logits.view(-1, self.num_labels_pos), labels_pos.view(-1))

            # Combine losses
            if (UNCERTAINTY_WEIGHTING):
                # Uncertainty-Based Weighting (Kendall et al., 2018)
                # Use the log variance parameters to weight the losses
                pos_term = torch.exp(-self.log_var_pos) * loss_pos + self.log_var_pos
                lid_term = torch.exp(-self.log_var_lid) * loss_lid + self.log_var_lid
                loss = pos_term.squeeze() + lid_term.squeeze()
            else:
                # Use the provided weights for each task
                loss = (LID_WEIGHT * loss_lid) + (POS_WEIGHT * loss_pos) 

        if not return_dict:
             output = (lid_logits, pos_logits)
             return ((loss,) + output) if loss is not None else output

        return {
            "loss": loss,
            "lid_logits": lid_logits,
            "pos_logits": pos_logits,
            # "loss_lid": loss_lid, # passing these is optional - only if want to track/log losses for each separately
            # "loss_pos": loss_pos,
        }