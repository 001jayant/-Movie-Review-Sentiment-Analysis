# SentimentScope — Transformer-Based Sentiment Analysis

SentimentScope is a Natural Language Processing (NLP) project that performs **binary sentiment classification** on movie reviews from the IMDB dataset.

The project implements a customized Transformer architecture called **DemoGPT** using PyTorch. The model processes movie reviews and predicts whether the sentiment expressed in each review is **Positive** or **Negative**.

The project was developed as an end-to-end machine learning workflow covering data loading, exploration, preprocessing, tokenization, Transformer implementation, training, validation, and final evaluation.

---

## Project Objective

The primary objective of this project is to build and evaluate a Transformer-based sentiment classification model using the IMDB movie review dataset.

The project aims to demonstrate how Transformer components such as:

* Token embeddings
* Positional embeddings
* Self-attention
* Multi-head attention
* Feed-forward networks
* Residual connections
* Layer normalization

can be combined to solve a real-world NLP classification problem.

---

## Project Results

The final trained model achieved:

**Test Accuracy: 76.11%**

The project requirement was to achieve greater than **75% test accuracy**.

| Metric                 |        Result |
| ---------------------- | ------------: |
| Required Test Accuracy |         > 75% |
| Achieved Test Accuracy |    **76.11%** |
| Requirement            | **Satisfied** |

The model was evaluated on the separate IMDB test dataset containing 25,000 reviews.

---

## Dataset

The project uses the **IMDB Movie Review Dataset**.

The dataset contains 50,000 labeled movie reviews:

* 25,000 training reviews
* 25,000 test reviews

The reviews are divided into two sentiment classes:

| Label | Sentiment |
| ----: | --------- |
|     0 | Negative  |
|     1 | Positive  |

### Dataset Split

The original 25,000 training reviews were shuffled and divided into:

| Dataset    | Samples |
| ---------- | ------: |
| Training   |  22,500 |
| Validation |   2,500 |
| Test       |  25,000 |

The test dataset was kept separate from the training and validation data for final evaluation.

---

## Dataset Structure

The project expects the IMDB dataset to be organized as follows:

```text
aclImdb/
├── train/
│   ├── pos/
│   ├── neg/
│   └── unsup/
│
└── test/
    ├── pos/
    └── neg/
```

The `unsup` directory is not used for supervised training.

---

## Project Structure

```text
SentimentScope_Project/
│
├── aclImdb/
│   ├── train/
│   │   ├── pos/
│   │   ├── neg/
│   │   └── unsup/
│   │
│   └── test/
│       ├── pos/
│       └── neg/
│
├── SentimentScope_Completed.ipynb
├── requirements.txt
├── README.md
│
└── sentimentscope_best.pt       # Optional trained checkpoint
```

---

## Technologies Used

* **Python**
* **PyTorch**
* **Hugging Face Transformers**
* **Pandas**
* **NumPy**
* **Matplotlib**
* **Scikit-learn**
* **Jupyter Notebook**

---

## Model Architecture

The project implements a customized Transformer model named **DemoGPT**.

The overall architecture is:

```text
                    Movie Review
                         │
                         ▼
               BERT Tokenizer
                         │
                         ▼
                    Token IDs
                         │
                         ▼
                 Token Embedding
                         │
                         +
                         │
              Positional Embedding
                         │
                         ▼
                Transformer Blocks
                         │
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
      Multi-Head Attention    Feed Forward
              │                     │
              └──────────┬──────────┘
                         │
                  Residual Connections
                         │
                  Layer Normalization
                         │
                         ▼
                 Final Representation
                         │
                         ▼
                  Classification Head
                         │
                 ┌───────┴───────┐
                 ▼               ▼
              Negative         Positive
                 0                1
```

---

## Transformer Components

### 1. Token Embeddings

The input token IDs are converted into dense vector representations using a PyTorch embedding layer.

### 2. Positional Embeddings

Positional embeddings provide information about the order of tokens in the review.

### 3. Multi-Head Self-Attention

The attention mechanism allows the model to identify relationships between different tokens in a review.

Multiple attention heads allow the model to learn different types of relationships simultaneously.

### 4. Feed-Forward Network

Each Transformer block contains a feed-forward network that further processes the representations generated by the attention mechanism.

### 5. Residual Connections

Residual connections help preserve information as it passes through the Transformer blocks and improve optimization.

### 6. Layer Normalization

Layer normalization helps stabilize the training process.

### 7. Classification Head

The final representation is passed through a linear classification layer that produces two output classes:

```text
0 → Negative
1 → Positive
```

---

## Tokenization

The project uses the Hugging Face:

```text
bert-base-uncased
```

tokenizer.

Each movie review is:

1. Tokenized into subword tokens
2. Truncated when necessary
3. Padded to a fixed sequence length
4. Converted into PyTorch tensors

The maximum sequence length used in the final experiment is:

```text
128 tokens
```

This produces input tensors with the shape:

```text
(batch_size, 128)
```

For example:

```text
Batch input shape: torch.Size([32, 128])
Batch labels shape: torch.Size([32])
```

---

## Data Exploration

Before training, the dataset was explored to understand its structure and characteristics.

The project includes:

* Dataset information
* Descriptive statistics
* Positive/negative sentiment distribution
* Review-length analysis
* Sample positive reviews
* Sample negative reviews

### Visualizations

At least two labeled visualizations were created:

1. **Sentiment Distribution**

   * Displays the number of positive and negative reviews.

2. **Review Length Distribution**

   * Shows the distribution of review lengths.

These visualizations provide an overview of the dataset before model training.

---

## PyTorch Dataset

A custom `IMDBDataset` class was implemented to convert the Pandas dataset into a format suitable for PyTorch.

The Dataset performs the following operations:

```text
Review
  ↓
Tokenizer
  ↓
Truncation
  ↓
Padding
  ↓
Tensor
```

The Dataset returns:

```text
(input_ids, label)
```

The project then uses PyTorch `DataLoader` objects for:

* Training
* Validation
* Testing

---

## Model Configuration

The final experiment used the following configuration:

| Parameter               |              Value |
| ----------------------- | -----------------: |
| Vocabulary Size         |             30,522 |
| Number of Classes       |                  2 |
| Embedding Dimension     |                128 |
| Context Size            |                128 |
| Transformer Layers      |                  4 |
| Attention Heads         |                  4 |
| Head Size               |                 32 |
| Dropout                 |                0.1 |
| Maximum Sequence Length |                128 |
| Batch Size              |                 32 |
| Epochs                  |                  3 |
| Optimizer               |              AdamW |
| Learning Rate           |             0.0003 |
| Loss Function           | Cross-Entropy Loss |

---

## Training

The model was trained using supervised learning.

For each batch, the following process was performed:

```text
Input Reviews
      ↓
Tokenization
      ↓
Transformer Model
      ↓
Classification Logits
      ↓
Cross-Entropy Loss
      ↓
Backpropagation
      ↓
AdamW Optimizer
      ↓
Updated Model Parameters
```

Validation accuracy was calculated after each training epoch to monitor model performance on unseen validation data.

---

## Evaluation

The model uses classification accuracy as its primary evaluation metric.

Accuracy is calculated as:

```text
Accuracy = (Correct Predictions / Total Predictions) × 100
```

The final model was evaluated using the independent IMDB test dataset containing 25,000 reviews.

---

## Final Result

The completed model achieved:

```text
76.11% Test Accuracy
```

This exceeded the required threshold:

```text
Required:  >75%
Achieved:   76.11%
```

Therefore, the model successfully met the project's required performance objective.

---

## Key Takeaways

### 1. Transformers are effective for sentiment classification

The customized Transformer architecture successfully learned useful sentiment patterns from movie reviews and achieved 76.11% accuracy on previously unseen test data.

### 2. Data preparation is essential

Tokenization, padding, truncation, dataset splitting, and batching are important steps in preparing natural-language data for deep learning.

### 3. Self-attention captures relationships between tokens

Multi-head self-attention allows the model to consider relationships between different parts of a review, which is important for understanding sentiment.

### 4. Validation provides an independent measure during training

Using a separate validation set makes it possible to monitor generalization during training while keeping the test set reserved for final evaluation.

---

## Model Checkpoint

The project can optionally save the trained model as:

```text
sentimentscope_best.pt
```

A checkpoint can contain:

* Model weights
* Model configuration
* Maximum sequence length
* Validation accuracy

Example:

```python
torch.save({
    "model_state_dict": model.state_dict(),
    "config": config,
    "max_length": MAX_LENGTH,
    "val_accuracy": val_accuracy
}, "sentimentscope_best.pt")
```

Saving the checkpoint makes it possible to reuse the trained model without retraining it.

---

## Inference

A reusable inference interface can be created to load the saved model and classify new reviews.

Example:

```python
predictor = SentimentScopePredictor(
    "sentimentscope_best.pt",
    tokenizer
)

reviews = [
    "This movie was absolutely fantastic!",
    "The story was boring and disappointing."
]

results = predictor.predict(reviews)

for result in results:
    print(result)
```

The inference interface can return:

```text
Positive
Negative
```

along with prediction confidence.

---

## Installation

### 1. Create a Virtual Environment

On Windows:

```bash
python -m venv .venv
```

Activate it:

```bash
.venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

If Jupyter is not already installed:

```bash
pip install jupyter ipykernel
```

---

## Running the Project

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
SentimentScope_Completed.ipynb
```

Run the notebook cells **from top to bottom**.

Running cells sequentially is important because later cells depend on variables, datasets, tokenizer objects, model configurations, and DataLoaders created earlier.

---

## Hardware

The project automatically detects whether a CUDA-compatible GPU is available.

```python
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
```

If CUDA is unavailable, the model runs on CPU.

Training on a GPU is recommended for larger experiments because Transformer models can be computationally expensive.

---

## Future Improvements

Several improvements can be explored in future versions of SentimentScope.

### Larger Model

Experiment with:

* Larger embedding dimensions
* More Transformer layers
* More attention heads

### Longer Context

Increase the maximum sequence length from 128 to 256 or higher to retain more information from long reviews.

### Training Improvements

Experiment with:

* Learning-rate scheduling
* Weight decay
* Gradient clipping
* More training epochs
* Early stopping
* Best-validation checkpointing

### Improved Classification

Different pooling strategies could be evaluated, such as:

* Mean pooling
* `[CLS]` token representation
* Attention-based pooling

### Inference Application

The trained model could be integrated into:

* A Flask API
* A FastAPI service
* A Streamlit application
* A web-based sentiment analysis interface

---

## Limitations

The current project has several limitations.

### Limited Sequence Length

Reviews longer than 128 tokens may be truncated, causing some information to be lost.

### Computational Cost

Increasing the Transformer size or sequence length significantly increases memory usage and training time.

### Complex Sentiment

Sarcasm, mixed opinions, ambiguous language, and complex sentence structures can make sentiment classification difficult.

### Model Capacity

The custom Transformer is relatively small compared with modern large language models, so it has limited capacity for representing complex language patterns.

---

## Conclusion

SentimentScope demonstrates a complete Transformer-based NLP workflow using the IMDB movie review dataset.

The project successfully covers:

* Data loading
* Data exploration
* Data visualization
* Text preprocessing
* Tokenization
* PyTorch Dataset and DataLoader
* Transformer implementation
* Self-attention
* Model training
* Validation
* Test evaluation
* Performance analysis

The final model achieved **76.11% test accuracy**, exceeding the required **75% accuracy target**.

The project provides a practical foundation for further experimentation with Transformer architectures, model optimization, checkpointing, and production-ready inference.

---

## Author

**Jayant Shende**

**Project:** SentimentScope — Transformer-Based Sentiment Analysis

**Technology:** Python, PyTorch, Hugging Face Transformers, NLP
