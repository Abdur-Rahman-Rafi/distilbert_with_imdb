# DistilBERT Sentiment Analysis with Uncertainty Estimation

## Overview

This project performs **sentiment analysis** on movie reviews using the **DistilBERT** language model and explores multiple techniques for measuring **model uncertainty**.

The model is evaluated on the **IMDB Movie Review Dataset**, and several uncertainty indicators are extracted to better understand prediction reliability.

---

## Objectives

* Perform sentiment classification using a pretrained DistilBERT model.
* Evaluate prediction confidence.
* Measure uncertainty using entropy.
* Analyze hidden-state representations.
* Study prediction stability across multiple runs.

---

## Model

**Model Used**

* `distilbert-base-uncased-finetuned-sst-2-english`

This is a lightweight transformer model fine-tuned for binary sentiment classification.

### Sentiment Labels

| Label | Meaning  |
| ----- | -------- |
| 0     | Negative |
| 1     | Positive |

---

## Dataset

**Dataset:** IMDB Movie Reviews

Loaded directly from the Hugging Face Datasets library:

```python
from datasets import load_dataset

dataset = load_dataset("imdb")
```

The experiment uses samples from the test split.

---

## Uncertainty Metrics

### 1. Confidence Score

The model outputs class probabilities using Softmax.

```python
confidence = max(probabilities)
```

Interpretation:

* High confidence → prediction is more certain.
* Low confidence → prediction is less certain.

---

### 2. Entropy

Entropy measures uncertainty in the probability distribution.

Formula:

[
H(p) = - \sum p(x)\log p(x)
]

Interpretation:

* Low entropy → model strongly favors one class.
* High entropy → model is unsure between classes.

---

### 3. Hidden-State Analysis

The model returns hidden states from every transformer layer.

The project extracts:

* Last hidden layer embeddings
* Embedding variance

```python
embedding_variance = torch.var(last_hidden)
```

Interpretation:

* Low variance → stable internal representations.
* High variance → potentially unstable or uncertain representations.

---

### 4. Multi-Run Stability Test

The same sentence is evaluated multiple times:

```python
"The movie was okay, not too bad but not amazing either."
```

The goal is to check whether predictions remain consistent across repeated executions.

Interpretation:

* Stable outputs → reliable prediction.
* Changing outputs → higher uncertainty.

---

## Project Workflow

### Step 1 – Install Dependencies

```bash
pip install transformers datasets torch
```

### Step 2 – Import Libraries

* PyTorch
* Transformers
* Hugging Face Datasets
* NumPy

### Step 3 – Detect GPU

The notebook checks GPU availability and automatically uses CUDA when available.

### Step 4 – Load DistilBERT

Tokenizer and model are loaded from Hugging Face.

### Step 5 – Load IMDB Dataset

Movie reviews and labels are retrieved from the test split.

### Step 6 – Define Entropy Function

A custom entropy function is implemented for uncertainty calculation.

### Step 7 – Prediction Function

The main prediction pipeline:

* Tokenization
* Forward pass
* Softmax probabilities
* Confidence calculation
* Entropy calculation
* Hidden-state extraction
* Embedding variance computation

### Step 8 – Label Mapping

Maps model outputs to:

* POSITIVE
* NEGATIVE

### Step 9 – Run Experiment

For each review:

* Display text sample
* Predict sentiment
* Compare with true label
* Report uncertainty metrics

### Step 10 – Stability Analysis

Repeated predictions are performed on an ambiguous sentence.

### Step 11 – Interpretation

A guide is printed to help interpret the uncertainty metrics.

---

## Example Output

```text
TRUE LABEL: POSITIVE
PREDICTED LABEL: POSITIVE

CONFIDENCE SCORE: 0.9845
ENTROPY: 0.0782
EMBEDDING VARIANCE: 0.234871
```

---

## Interpretation Guide

### High Confidence + Low Entropy

* Model is highly certain.
* Prediction is likely reliable.

### Low Confidence + High Entropy

* Model is uncertain.
* Prediction should be treated cautiously.

### High Embedding Variance

* Internal representations are unstable.
* May indicate uncertainty in understanding the input.

### Prediction Changes Across Runs

* Suggests low model stability.
* Indicates increased uncertainty.

---

## Technologies Used

* Python
* PyTorch
* Hugging Face Transformers
* Hugging Face Datasets
* DistilBERT
* CUDA (optional)

---

## Educational Purpose

This project demonstrates how uncertainty can be analyzed in a pretrained Large Language Model (LLM) without prompt engineering by combining:

* Confidence estimation
* Entropy analysis
* Hidden-state inspection
* Stability testing

It serves as a simple introduction to uncertainty quantification in transformer-based NLP systems.

---

## Future Improvements

* Monte Carlo Dropout uncertainty estimation
* Calibration metrics (ECE, Brier Score)
* Attention-based uncertainty analysis
* Visualization of hidden-state embeddings
* Comparison with larger transformer models (BERT, RoBERTa, DeBERTa)

---

## Author

Small LLM & Uncertainty Analysis Project

Sentiment Analysis using DistilBERT with Multi-Granular Uncertainty Signals.
