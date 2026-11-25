# NLP Document Classifier

A deep learning-based document classification system built with **PyTorch**, exploring NLP preprocessing, custom **BiLSTM** networks, Transformer architectures, and **BERT fine-tuning**.

## Project Overview

The goal of this project is to build a system that can analyse the text in a document and automatically assign it to an appropriate category. The project will explore two deep-learning approaches:

1. **Custom BiLSTM classifier** — build and train a neural network using PyTorch, word embeddings, and a Bidirectional LSTM.
2. **Transformer-based classifier** — fine-tune a pretrained Transformer model such as BERT for document classification.

The two approaches will be evaluated and compared to understand their performance, training requirements, and practical trade-offs.

## Objectives

- Learn PyTorch fundamentals, including tensors, autograd, and neural network training.
- Understand NLP preprocessing, tokenisation, vocabulary construction, and word embeddings.
- Build a document classifier using a custom BiLSTM architecture.
- Understand Transformer models, attention, and BERT.
- Fine-tune a pretrained language model for text classification.
- Evaluate and compare both approaches using standard classification metrics.
- Develop an application that accepts documents and predicts their categories.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| PyTorch | Deep learning framework |
| Hugging Face Transformers | Pretrained Transformer models and tokenisers |
| scikit-learn | Dataset utilities and evaluation metrics |
| NumPy and Pandas | Data manipulation |
| NLTK / spaCy | NLP preprocessing |
| PyMuPDF | Extracting text from PDF documents |
| Streamlit | Building the user-facing application |
| Matplotlib | Visualising results |

## Project Workflow

1. **Dataset collection:** obtain and inspect a labelled text-classification dataset.
2. **Exploratory data analysis:** examine categories, class distribution, document lengths, and data quality.
3. **Preprocessing:** clean text, tokenise documents, and prepare model inputs.
4. **BiLSTM model:** create embeddings, build a Bidirectional LSTM classifier, and train it with PyTorch.
5. **Transformer model:** load a pretrained BERT-family model and fine-tune it on the classification dataset.
6. **Evaluation:** compare model performance and computational requirements.
7. **Application:** accept a document, extract its text, and return a predicted category.

## Dataset

Candidate datasets include:

- **BBC News:** news articles grouped into topic categories.
- **AG News:** news headlines and descriptions across four categories.
- **20 Newsgroups:** documents from a range of discussion groups.

The selected dataset and its source, licence, and preparation steps will be documented in `data/README.md`. Keep raw datasets in `data/raw/` and processed datasets in `data/processed/`. Dataset files should not be committed to Git unless their licence and size make that appropriate.

## Evaluation

The models will be evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Training time
- Inference time

The comparison will help assess not only predictive performance but also the practical cost of each approach.

## Repository Structure

```text
nlp-document-classifier/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
├── notebooks/
│   ├── 01_pytorch_fundamentals.ipynb
│   ├── 02_tensor_operations.ipynb
│   ├── 03_autograd_and_gradients.ipynb
│   ├── 04_neural_network_fundamentals.ipynb
│   ├── 05_nlp_preprocessing.ipynb
│   ├── 06_word_embeddings.ipynb
│   ├── 07_bilstm_classifier.ipynb
│   ├── 08_transformer_fundamentals.ipynb
│   └── 09_bert_classifier.ipynb
├── src/
│   ├── data/
│   │   ├── load_data.py
│   │   └── preprocess.py
│   ├── models/
│   │   ├── embeddings.py
│   │   ├── bilstm.py
│   │   └── bert.py
│   ├── training/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── predict.py
│   └── utils/
│       ├── metrics.py
│       └── config.py
├── experiments/
│   ├── bilstm/
│   └── bert/
├── models/
├── reports/
│   ├── figures/
│   └── model_comparison.md
├── app/
│   └── app.py
└── tests/
    └── test_preprocessing.py
```

## Learning Focus

This repository is also a hands-on learning journey. Each notebook introduces a concept and connects it to the implementation:

- PyTorch tensors and tensor operations
- Autograd, computational graphs, and gradients
- Neural network fundamentals and optimisation
- NLP preprocessing and tokenisation
- Word embeddings
- Recurrent neural networks and BiLSTM
- Attention and Transformer architecture
- BERT and transfer learning
- Model training, evaluation, and inference

## Current Status

**In development.** The repository structure and learning notebooks have been set up. Model implementation, experiments, evaluation, and the application will be added progressively.

## Future Improvements

- Experiment with additional pretrained language models.
- Add support for more document formats and categories.
- Improve handling of long documents.
- Add model explainability and confidence scores.
- Package the application for deployment.

## Author

Developed as a practical learning project in deep learning and Natural Language Processing (NLP).
