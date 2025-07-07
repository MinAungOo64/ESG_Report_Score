# ESG Score Prediction from Sustainability Reports

This project uses a custom regression model based on [Longformer](https://huggingface.co/allenai/longformer-base-4096) to predict ESG (Environmental, Social, Governance) scores from long-form sustainability reports of S&P 500 companies.  
The dataset trained on can be found here: https://www.kaggle.com/datasets/jaidityachopra/esg-sustainability-reports-of-s-and-p-500-companies/data  


---

## Model Overview

- **Architecture**: `Longformer + Linear Regression Head`
- **Task**: Multi-output regression
- **Outputs**:
  - `e_score`: Environmental score  
  - `s_score`: Social score  
  - `g_score`: Governance score  
  - `total_score`: Combined ESG score
- **Input**: Preprocessed sustainability report text (~10k words)

### ESG Score methodology
Information on kaggle stated that ESG scores were obtained from Sustainalytics. However there were no clear information on the methodology of the scoring. Additionally there were no individual e_score, s_core, g_score, only a total_score was found. More information maybe found in the individual reports that are currently unavailable.  
Regardless, the model will be trained to predict all four outputs for experimentation eventhough total_score maybe different from the 3 individual scores combined.

Extracted from Sustainalytics, below shows the scale of ESG ratings, the higher the score, the higher the ESG risks. A lower score is preferred.  
Negligible: 0-10, Low: 10-20, Medium: 20-30, High: 30-40, Severe: 40+

---

## Files Included

- esg_tokenized - Contains tokenized data
- LongFormer_ESG_Score - Contains trained model weights in `pytorch_model.bin`
- `Data Processing and Tokenization.ipynb` - Data preprocessing
- `Training the Model.ipynb` - Model training
- `Testing the model.ipynb` - Model testing
- `README.md` — Project overview

---

## Important Steps

Below I will outline some of the important steps taken to train the model for efficiency and speed.

### Apache Arrow Format
With the datasets library, we load our data in Apache Arrow format. In this format, the data is not loaded into memory which speeds up functions like batching, filtering and .map() functions.

```python
from datasets import DatasetDict, load_dataset

# Load full dataset
full_dataset = load_dataset("csv", data_files="esg_dataset.csv")["train"]
```

### Tokenizing
The tokenizer used is from the longformer-base-4096 model which can input up to 4096 tokens which is ideal for ESG reports that contains more than 10k+ words.

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
```
Before passing into .map() function to tokenize every row of the data, a tokenize_and_split() function is defined.  

`truncation=True`	- If the report is too long, it will be cut off at max_length.  
`max_length=1024`	- The tokenizer will output at most 1024 tokens per chunk. I set it at 1024 tokens instead of the max 4096 tokens becauses my GPU has limited vram (4GB). You can increase this value if your GPU is stronger.  
`return_overflowing_tokens=True` - If a report is longer than 1024 tokens, split it into multiple overlapping chunks.  
`stride=256` -	When splitting into chunks, overlap each chunk by 256 tokens with the previous one (helps preserve context between chunks).  

```python
def tokenize_and_split(examples):
    result = tokenizer(
        examples["report"],
        truncation=True,
        max_length=1024,
        return_overflowing_tokens=True,
        stride=256
    )
```

### Data Loader
The data loader loads data into the model with custom settings. It has dynamic padding where tokenzied inputs are padded to the longest inputs in each batch for efficiency.

```python
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

# For dynamic padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Prepare the data in batch size of 2 with dynamic padding, shuffles at each epoch
train_dataloader = DataLoader(
    tokenized_dataset["train"], shuffle=True, batch_size=2, collate_fn=data_collator
)
``` 

### Custom Regression Model
Normally, the longformer-base-4096 modelis is used for classification task such as sentiment analysis with predefined labels. As such the model is modified for regression in order to predict a continous variable like ESG score.

```python
from transformers import LongformerModel
import torch.nn as nn

class LongformerForRegression(nn.Module):
    def __init__(self):
        super().__init__()
        # Load the pretrained Longformer model (handles up to 4096 tokens)
        self.longformer = LongformerModel.from_pretrained("allenai/longformer-base-4096")
        
        # Dropout for regularization (prevents overfitting)
        self.dropout = nn.Dropout(0.1)
        
        # Final regression layer to predict 4 scores: e_score, s_score, g_score, total_score
        self.regressor = nn.Linear(self.longformer.config.hidden_size, 4)

    def forward(self, input_ids, attention_mask):
        # Forward pass through Longformer
        outputs = self.longformer(input_ids=input_ids, attention_mask=attention_mask)
        
        # Try to use pooled output (if available)
        pooled_output = outputs.pooler_output
        
        # If pooled output is None (some models don't provide it), use mean pooling over all tokens
        if pooled_output is None:
            pooled_output = outputs.last_hidden_state.mean(dim=1)  # shape: (batch_size, hidden_size)
        
        # Apply dropout
        pooled_output = self.dropout(pooled_output)
        
        # Pass through regression head to get 4 continuous outputs
        return self.regressor(pooled_output)
```

### Training Loop
We set the loss function as Mean-Squared-Error so that the model will update the weights to reduce the MSE after each forward pass.  
Note: With my GPU RTX 3050ti, it took at total of 18hrs to fully train this model.

#### Forward Pass
At the beginning the model weights are randomnized.  
The training loop contains 3 epochs which means the entire tokenized dataset will be passed through the model 3 times, once at each epoch.  
In each epoch, the data loader loads a batch containing 2 tokenized inputs into the model with the initial randomnized weights and outputs a score.  
The score is comapred with the true score to calculate the loss: MSE. This is known as the forward pass.

#### Backward Pass
With loss calculated, the model attempts to reduce the loss by modifying the weights with an optimizer.  
The smaller the loss, the closer the predicted score is to the true score. This is known as the backward pass

#### Validation
After each epoch, we test the model on the validation set to calculate the loss. If the loss is not reduced significantly after consecutive epochs, the model stops early to prevent the model overfitting to the train data. This is particularly useful when you plan to have higher number of epochs. 

### Saving the weights
After training, we save the weights in `pytorch_model.bin` for future use. We also save the tokenizer that was used in data preprocessing just in case we forget.

```python
import torch

# Save model weights
output_dir = "C:/Users/steve/HuggingFace Models/LongFormer_ESG_Score"
torch.save(model.state_dict(), f"{output_dir}/pytorch_model.bin")

# Save tokenizer as usual (since it's from Hugging Face)
tokenizer.save_pretrained(output_dir)
```

### Testing the model
Loading the saved weights we test the model on unseen data.

---

## Model Results
The test loss MSE is 27.5613. The RMSE is 5.25.  
This means on average, the prediction score is off by 5.25.  
This is decent considering loss is calculated with the average loss of all four scores.  
The prediction can definitely be improved if the model is only trained to predict the 3 score individually then calculated the total score.  
If we are only concerned about the total score then we can train the model to predict just the total score and the results will be much better.

---
