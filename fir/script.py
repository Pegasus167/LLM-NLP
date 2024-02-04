from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle
import pandas as pd

# Load your dataset
df = pd.read_csv("firfinal26.csv")
# Replace X and y with your actual data
X = df['New Description']
y = df['section']

# Use LabelEncoder to convert 'IPC_section' to integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Save the label_encoder to a file using pickle
with open('label_encoder.pkl', 'wb') as le_file:
    pickle.dump(label_encoder, le_file)

# Load pre-trained BERT model and tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(label_encoder.classes_))

# Tokenize and format input text for the model
inputs = tokenizer(X.tolist(), truncation=True, padding=True, return_tensors="pt", max_length=512)

# Create DataLoader
dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], torch.tensor(y_encoded))
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Training parameters
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
criterion = torch.nn.CrossEntropyLoss()

# Training loop
num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    for batch in dataloader:
        input_ids, attention_mask, labels = batch

        # Convert labels to torch.LongTensor explicitly
        labels = labels.long()

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

# Now, you can use the trained model for prediction

# Later, load the label_encoder from the saved file
with open('label_encoder.pkl', 'rb') as le_file:
    loaded_label_encoder = pickle.load(le_file)