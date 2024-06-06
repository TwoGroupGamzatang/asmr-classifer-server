from torch.utils.data import Dataset

class ArticleDataset(Dataset):
    def __init__(self, texts, tokenizer, labels):
        self.tokenizer = tokenizer
        self.texts = texts.tolist()
        self.targets = labels.tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        target = self.targets[idx]

        # Tokenize the text
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()

        return input_ids, attention_mask, target
