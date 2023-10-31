from transformers import BartForConditionalGeneration, BartTokenizer, TrainingArguments, Trainer
import datasets
import pandas as pd
import torch

# Load your custom dataset (assuming it's a CSV file)
csv_file_path = r"C:\Users\kumar\OneDrive\Desktop\ContentCreation\modify_samsum_dataset\test.csv"  # Replace with your CSV file path
data = pd.read_csv(csv_file_path)

# Convert the pandas DataFrame to a Hugging Face Dataset
dataset = datasets.Dataset.from_pandas(data)

# Rename columns to avoid conflicts
dataset = dataset.rename_column("dialogue", "input_text")
dataset = dataset.rename_column("summary", "target_text")

# Initialize the model and tokenizer
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

# Tokenize and format the dataset with maximum sequence length
def tokenize_function(examples):
    inputs = tokenizer(examples["input_text"], truncation=True, padding="max_length", max_length=128, return_tensors="pt")
    outputs = tokenizer(examples["target_text"], truncation=True, padding="max_length", max_length=128, return_tensors="pt")
    return {"input_ids": inputs["input_ids"], "labels": outputs["input_ids"]}

# Tokenize the dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)
# Define a custom Trainer class with a compute_loss method
class MyTrainer(Trainer):
    def compute_loss(self, model, inputs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1))
        return loss

# Define training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=2,
    evaluation_strategy="steps",
    save_steps=10_000,
    output_dir="./custom_bart_model",
)

# Initialize the Trainer and fine-tune the model
trainer = MyTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
)

trainer.train()