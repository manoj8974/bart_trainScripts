from transformers import BartTokenizer, BartForConditionalGeneration
from torch.utils.data import DataLoader, Dataset
import torch

# Assuming you have a custom dataset in the form of lists of source_text and target_text
source_texts = ["Ram is a good boy", "sita is a good girl"]  # List of source texts
target_texts = ["Ram is good","Sita is good"]  # List of corresponding target summaries

# Initialize the BART tokenizer and model
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

# Tokenize the input texts and target summaries
inputs = tokenizer(source_texts, truncation=True, padding='longest', return_tensors='pt')
targets = tokenizer(target_texts, truncation=True, padding='longest', return_tensors='pt')

# Create a custom dataset
class CustomDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs["input_ids"])

    def __getitem__(self, idx):
        return {
            'input_ids': self.inputs['input_ids'][idx],
            'attention_mask': self.inputs['attention_mask'][idx],
            'labels': self.targets['input_ids'][idx],
        }

dataset = CustomDataset(inputs, targets)

# DataLoader for batching and shuffling
train_loader = DataLoader(dataset, batch_size=2, shuffle=True)

# Define the optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)

# Training loop
num_epochs = 3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.train()

for epoch in range(num_epochs):
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        print(f"Epoch: {epoch + 1}, Loss: {loss.item()}")

# Save the trained model
model.save_pretrained("./custom_bart_model")

""" 
This code assumes you have pre-processed your custom dataset into lists of source texts and target summaries. It tokenizes the input texts and target summaries using the BART tokenizer and creates a custom PyTorch dataset for training. The model is then trained using a DataLoader in a basic training loop over a specified number of epochs.

Ensure you replace source_texts, target_texts, and specify your path_to_save_model. Additionally, make sure to handle your dataset preprocessing, padding, and tokenization as required by your specific use case.

Adjust the batch size, learning rate, and other hyperparameters based on the available computational resources and the characteristics of your dataset. Also, consider saving and loading checkpoints for model training continuity if required.

"""
