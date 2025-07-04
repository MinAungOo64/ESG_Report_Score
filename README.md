# 📊 ESG Score Prediction from Sustainability Reports

This project uses a custom regression model based on [Longformer](https://huggingface.co/allenai/longformer-base-4096) to predict ESG (Environmental, Social, Governance) scores from long-form sustainability reports of S&P 500 companies.

---

## 🧠 Model Overview

- **Architecture**: `Longformer + Linear Regression Head`
- **Task**: Multi-output regression
- **Outputs**:
  - `e_score`: Environmental score  
  - `s_score`: Social score  
  - `g_score`: Governance score  
  - `total_score`: Combined ESG score
- **Input**: Preprocessed sustainability report text (~10k words)

---

## 📁 Files Included

- `pytorch_model.bin` — Trained model weights (PyTorch)
- Tokenizer files — From `allenai/longformer-base-4096`
- `README.md` — Project overview
- *(Optional)* `config.json` — If wrapping with Hugging Face `PreTrainedModel`

---

## 🚀 How to Use

```python
from transformers import LongformerTokenizer
import torch
from your_module import LongformerForRegression  # Define your model class

# Load tokenizer
tokenizer = LongformerTokenizer.from_pretrained("your-username/longformer-esg-regressor")

# Load model
model = LongformerForRegression()
model.load_state_dict(torch.load("pytorch_model.bin"))
model.eval()

# Example input
inputs = tokenizer("Sustainability report text here...", return_tensors="pt", truncation=True, max_length=4096)
with torch.no_grad():
    predictions = model(**inputs)
print(predictions)  # e_score, s_score, g_score, total_score
