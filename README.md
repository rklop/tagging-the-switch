# COSC243 Final Project: Multilingual NLP with Transformer Models

## Project Overview

This project implements and evaluates various training strategies for multilingual Natural Language Processing (NLP) tasks, specifically **Language Identification (LID)** and **Part-of-Speech (POS) Tagging**. The implementation explores three distinct training paradigms:

1. **Single-Task Training**: Individual models trained separately for each task
2. **Sequential Training**: Models trained sequentially on one task, then fine-tuned on another
3. **Multi-Task Training**: Single model trained simultaneously on both tasks with various weighting strategies

## Architecture & Technical Details

### Model Architecture
- **Base Model**: XLM-RoBERTa (`xlm-roberta-base`) with RemBERT configuration
- **Task Heads**: Separate classification layers for LID and POS tagging
- **Uncertainty Weighting**: Implementation of Kendall et al. (2018) uncertainty-based loss weighting
- **Multi-Task Framework**: Shared encoder with task-specific classification heads

### Training Strategies

#### Single-Task Training (`train_single.py`)
- Independent training for each task (LID or POS)
- Task-specific hyperparameter optimization
- Baseline performance establishment

#### Sequential Training (`train_sequential.py`)
- Phase 1: Train on primary task (LID or POS)
- Phase 2: Fine-tune on secondary task using Phase 1 weights
- Order-dependent performance analysis (LID→POS vs POS→LID)

#### Multi-Task Training (`train_multitask.py`)
- **Unweighted**: Equal loss contribution from both tasks
- **Weighted**: Configurable task-specific loss weights
- **Uncertainty-Based**: Dynamic loss weighting using learned uncertainty parameters

### Data Processing Pipeline

#### Dataset Structure
```
data/
├── LID/
│   ├── train.conll
│   ├── dev.conll
│   └── test.conll
└── POS/
    ├── train.conll
    ├── dev.conll
    └── test.conll
```

#### Data Format
- **CoNLL Format**: Standard NLP annotation format
- **Tokenization**: Hugging Face tokenizer integration
- **Label Mapping**: Dynamic label-to-ID and ID-to-label conversion
- **Sequence Length**: Configurable maximum sequence length (default: 128)

## Installation & Setup

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM

### Dependencies
```bash
pip install torch transformers datasets evaluate scikit-learn numpy pandas
```

### Environment Setup
```bash
# Clone repository
git clone <repository-url>
cd COSC243-FP

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

### Model Configuration (`src/config/config.py`)
```python
# Model Selection
MODEL_NAME = "xlm-roberta-base"
MODEL_CONFIG = RemBertConfig

# Training Hyperparameters
LEARNING_RATE = 5e-6
NUM_TRAIN_EPOCHS = 1
PER_DEVICE_TRAIN_BATCH_SIZE = 32
PER_DEVICE_EVAL_BATCH_SIZE = 32
WEIGHT_DECAY = 0.01
MAX_SEQUENCE_LENGTH = 128

# Task Weights (Multi-Task Training)
LID_WEIGHT = 1.0
POS_WEIGHT = 1.0
UNCERTAINTY_WEIGHTING = False
```

### Available Models
- **RemBERT**: `google/rembert` with `RemBertConfig`
- **XLM-RoBERTa**: `xlm-roberta-base` or `xlm-roberta-large` with `XLMRobertaConfig`
- **mBERT**: `bert-base-multilingual-cased` with `BertConfig`

## Usage

### Single-Task Training
```bash
python src/train_single.py --task pos  # or lid
```

### Sequential Training
```bash
python src/train_sequential.py
# Configure task order in config.py
```

### Multi-Task Training
```bash
python src/train_multitask.py
# Adjust weighting strategy in config.py
```

## Training Pipeline

### Data Loading & Preprocessing
1. **Dataset Loading**: CoNLL file parsing and validation
2. **Tokenization**: Hugging Face tokenizer with padding/truncation
3. **Label Processing**: Dynamic label mapping and validation
4. **Data Collation**: Batch preparation for training

### Training Process
1. **Model Initialization**: Pre-trained model loading with task-specific heads
2. **Training Arguments**: Hugging Face Trainer configuration
3. **Loss Computation**: Task-specific or combined loss calculation
4. **Evaluation**: Per-epoch validation with configurable metrics
5. **Model Saving**: Checkpoint and final model preservation

### Evaluation Metrics
- **F1 Score**: Primary evaluation metric for single-task and sequential training
- **Combined F1**: Multi-task evaluation metric
- **Per-Task Metrics**: Individual task performance analysis
- **Cross-Task Analysis**: Performance comparison across training strategies

## Model Outputs

### Saved Models
```
model/
├── single_task_pos-final/
├── single_task_lid-final/
├── sequential_pos_lid-final/
├── sequential_lid_pos-final/
├── multitask_unweighted-final/
└── multitask_weighted-final/
```

### Training Artifacts
```
saves/
├── sequential_phase1_pos/
├── sequential_phase1_lid/
└── multitask_training/
```

## Performance Analysis

### Evaluation Framework
- **Cross-Validation**: Robust performance assessment
- **Task Transfer**: Sequential training effectiveness analysis
- **Multi-Task Efficiency**: Resource utilization comparison
- **Uncertainty Quantification**: Confidence estimation in predictions

### Expected Outcomes
- **Single-Task**: Baseline performance benchmarks
- **Sequential**: Task transfer learning effectiveness
- **Multi-Task**: Joint learning efficiency and performance trade-offs

## Technical Implementation Details

### Custom Model Architecture
```python
class MultitaskTokenClassification(PreTrainedModel):
    def __init__(self, config):
        # Shared encoder
        self.model = AutoModel.from_config(config)
        
        # Task-specific classifiers
        self.lid_classifier = nn.Linear(config.hidden_size, num_labels_lid)
        self.pos_classifier = nn.Linear(config.hidden_size, num_labels_pos)
        
        # Uncertainty parameters
        self.log_var_pos = nn.Parameter(torch.zeros(1))
        self.log_var_lid = nn.Parameter(torch.zeros(1))
```

### Loss Functions
- **Cross-Entropy Loss**: Standard classification loss
- **Uncertainty Weighting**: `exp(-log_var) * loss + log_var`
- **Task Weighting**: Configurable per-task loss contribution

### Training Optimizations
- **Mixed Precision**: FP16 training when CUDA available
- **Gradient Accumulation**: Effective batch size management
- **Early Stopping**: Best model preservation
- **Learning Rate Scheduling**: Adaptive learning rate adjustment

## Research Contributions

### Novel Aspects
1. **Uncertainty-Based Weighting**: Implementation of Kendall et al. (2018) methodology
2. **Sequential vs. Multi-Task Comparison**: Comprehensive training strategy evaluation
3. **Multilingual Task Integration**: Joint LID and POS tagging optimization
4. **Dynamic Loss Balancing**: Adaptive task weighting strategies

### Experimental Design
- **Controlled Variables**: Model architecture, dataset, evaluation metrics
- **Independent Variables**: Training strategy, task order, weighting method
- **Dependent Variables**: F1 scores, training time, resource utilization

## Future Work

### Potential Extensions
1. **Additional Languages**: Extended multilingual support
2. **Task Expansion**: Named Entity Recognition (NER) integration
3. **Advanced Weighting**: Meta-learning for optimal task weights
4. **Model Compression**: Knowledge distillation and quantization

### Research Directions
1. **Cross-Lingual Transfer**: Zero-shot performance analysis
2. **Domain Adaptation**: Cross-domain generalization studies
3. **Efficiency Analysis**: Training time and resource optimization
4. **Interpretability**: Attention visualization and analysis

## Troubleshooting

### Common Issues
1. **CUDA Out of Memory**: Reduce batch size or sequence length
2. **Label Mismatch**: Verify dataset format and label consistency
3. **Model Loading Errors**: Check model name and configuration compatibility
4. **Training Instability**: Adjust learning rate and weight decay

### Performance Optimization
1. **Batch Size Tuning**: Balance memory usage and training efficiency
2. **Learning Rate Scheduling**: Implement warmup and decay strategies
3. **Mixed Precision**: Enable FP16 for faster training
4. **Gradient Clipping**: Prevent gradient explosion

## Contributing

### Development Guidelines
1. **Code Style**: Follow PEP 8 conventions
2. **Documentation**: Comprehensive docstrings and comments
3. **Testing**: Unit tests for critical functions
4. **Version Control**: Descriptive commit messages

### Project Structure
```
COSC243-FP/
├── src/
│   ├── config/          # Configuration management
│   ├── utils/           # Utility functions and models
│   └── training/        # Training scripts
├── data/                # Dataset storage
├── model/               # Saved models
├── saves/               # Training artifacts
└── docs/                # Documentation
```

## License

This project is developed for educational purposes as part of COSC243 coursework. Please refer to the course syllabus for usage and distribution guidelines.

## Acknowledgments

- **Hugging Face**: Transformers library and model implementations
- **PyTorch**: Deep learning framework
- **Research Community**: Kendall et al. (2018) uncertainty weighting methodology
- **COSC243 Instructors**: Project guidance and technical support

## Contact

For questions or issues related to this project, please contact the development team or refer to the course instructor.

---

**Note**: This project represents a comprehensive exploration of multilingual NLP training strategies and serves as a foundation for advanced research in cross-lingual language understanding and multi-task learning.
