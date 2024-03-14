import json
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        """
        CustomDataset class to load training data.

        Args:
            data (list): List of dictionaries containing training data.
            tokenizer (GPT2Tokenizer): Tokenizer for encoding the data.
            max_length (int, optional): Maximum length of the input and output sequences. Defaults to 512.
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

        if tokenizer.pad_token_id is None:
            tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns the item at the given index.

        Args:
            idx (int): Index of the item.

        Returns:
            dict: Dictionary containing the input_ids, attention_mask, and labels.
        """
        user_content = self.data[idx]['content']
        assistant_content = self.data[idx+1]['content'] if idx < len(self.data) - 1 else ""

        input_encoding = self.tokenizer(user_content, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        output_encoding = self.tokenizer(assistant_content, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")

        return {
            "input_ids": input_encoding.input_ids.flatten(),
            "attention_mask": input_encoding.attention_mask.flatten(),
            "labels": output_encoding.input_ids.flatten()
        }


def train_model(train_dataset, model, tokenizer, batch_size=4, num_epochs=3, learning_rate=1e-5):
    """
    Trains the GPT-2 model.

    Args:
        train_dataset (CustomDataset): CustomDataset object containing the training data.
        model (GPT2LMHeadModel): GPT-2 model to be trained.
        tokenizer (GPT2Tokenizer): Tokenizer for encoding the data.
        batch_size (int, optional): Batch size for training. Defaults to 4.
        num_epochs (int, optional): Number of epochs for training. Defaults to 3.
        learning_rate (float, optional): Learning rate for training. Defaults to 1e-5.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    for epoch in range(num_epochs):
        total_loss = 0.0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch + 1}/{num_epochs}')
        for step, batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': total_loss / (step + 1)})  

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")


with open('train.json', 'r') as f:
    training_data = json.load(f)


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model_config = GPT2Config.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2", config=model_config)

train_dataset = CustomDataset(training_data, tokenizer)

train_model(train_dataset, model, tokenizer)

output_dir = "./gpt2-small-chatlyrics"
model.save_pretrained(output_dir)
