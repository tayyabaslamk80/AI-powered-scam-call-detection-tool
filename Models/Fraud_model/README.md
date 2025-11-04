---
language: en
license: apache-2.0
base_model: distilbert-base-uncased
tags:
  - text-classification
  - fraud-detection
  - transformer
  - distilbert
  - huggingface
pipeline_tag: text-classification
widget:
  - text: "We require an urgent refund for the suspicious transaction on our account."
---

# 🕵️‍♂️ Fraud Model Aura (KS-Vijay)

This model uses **DistilBERT** to classify whether a given grievance or complaint text contains **fraudulent intent or behavior**. It is trained as part of an intelligent **Grievance Redressal Platform** to auto-detect fraud-related issues in startup complaints.

## 🧠 Use Case

Detects if the complaint relates to fraud:
- `Fraud`
- `Legitimate`

This helps startups or service providers to **flag, escalate, or triage suspicious reports** quickly.

## 🔍 Model Summary

- **Model Type:** Text Classification
- **Architecture:** DistilBERT (uncased)
- **Output Labels:** `Fraud`, `Legitimate`
- **Weights Format:** `safetensors`
- **Dataset:** Custom (based on complaints.csv)
- **Training Framework:** PyTorch using 🤗 `transformers`

## 📥 Example Input

```text
"I think someone is misusing our company’s KYC information to open fake accounts."
