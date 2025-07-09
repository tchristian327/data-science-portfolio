# ArXiv Title Generator with T5

## Project Overview
This project fine-tunes a T5-small transformer model to generate academic paper titles from ArXiv abstracts. The model is trained on a large subset of metadata records to learn how to compress technical abstracts into short, informative titles.

This project demonstrates:
- End-to-end text summarization using Hugging Face Transformers and TensorFlow
- Sequence-to-sequence modeling with real-world data
- Evaluation using ROUGE-L to assess title quality

## Dataset
- Source: arXiv Open Access metadata snapshot (`arxiv-metadata-oai-snapshot.json`)
- Sample Size: 200,000 records
- Fields Used:
  - `abstract` (input text)
  - `title` (target text)

## Tools & Libraries
- Python, Pandas, NumPy
- Hugging Face Transformers and Datasets
- TensorFlow / Keras
- ROUGE-L evaluation metric via keras-hub

## Model & Training Details
- Model checkpoint: `t5-small` (pretrained)
- Training objective: Abstract → Title
- Max input length: 384 tokens
- Max target length: 48 tokens
- Training set: 90%, Test set: 10%
- Batch size: 8
- Optimizer: Adam
- Evaluation metric: ROUGE-L F1 score (on a held-out sample)

## Workflow
1. Load and preprocess ArXiv metadata from JSON
2. Rename and split columns for summarization task
3. Tokenize with task prefix (`summarize:`) and pad/truncate sequences
4. Train using TensorFlow’s Keras API and Hugging Face Datasets
5. Evaluate using ROUGE-L with a custom Keras callback
6. Deploy inference pipeline with Hugging Face’s `pipeline` API

## Example Inference
After training, the model can generate titles based on unseen abstracts. Example:

```python
summarizer("We propose a novel transformer-based approach to...")
# Output: "A Transformer-Based Approach to..."
