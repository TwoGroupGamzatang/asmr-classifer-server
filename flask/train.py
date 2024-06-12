from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from datetime import datetime
from model import ArticleClassifier
from ArticleDataset import ArticleDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import BertTokenizer, AdamW
from utils import *
from torch.nn import CrossEntropyLoss
import copy


def filter_and_encode_labels(row, label_mapping, user_preferences):
    tags = [row['tag_1'], row['tag_2'], row['tag_3']]
    filtered_tags = [tag for tag in tags if tag in user_preferences]
    labels = [label_mapping[tag] for tag in filtered_tags if tag in label_mapping]
    return labels

def article_in_preferences(row, user_preferences):
    tags = [row['tag_1'], row['tag_2'], row['tag_3']]
    return any(tag in user_preferences for tag in tags)

def load_training_data(userID):
    # Assuming get_user_preferences is already defined and implemented
    user_preferences, label_mapping = get_user_preferences(userID)
    article_df = pd.read_excel('final_articles.xlsx')
    
    # Filter articles based on user preferences
    personal_article_df = article_df[article_df.apply(article_in_preferences, axis=1, user_preferences=user_preferences)]
    
    # Retain only the tags that match user preferences and encode labels
    personal_article_df['labels'] = personal_article_df.apply(filter_and_encode_labels, axis=1, label_mapping=label_mapping, user_preferences=user_preferences)
    
    summarized_text = personal_article_df['summarized']
    labels = personal_article_df['labels']
    print(labels)
    print(type(labels))

    return summarized_text, labels

def load_personal_training_data(userID):
    # Assuming get_user_preferences is already defined and implemented
    user_preferences, label_mapping = get_user_preferences(userID)
    data_path = os.path.join("user_data", f'{userID}.json')
    personal_article_df = pd.read_json(data_path, orient='records', lines=True)

    summarized_text = personal_article_df['summarized_text']
    labels = personal_article_df['labels'].apply(lambda x: eval(x) if isinstance(x, str) else x)  # Ensure labels are lists

    print(labels)
    print(type(labels))

    return summarized_text, labels


def train_save_personal_classifier(summarized_text, labels, tokenizer, user_preferences, userID):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_save_path = f'{userID}.pth'

    if os.path.exists(model_save_path):
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f'{userID}_{current_time}.pth'
        os.rename(model_save_path, backup_path)

    num_classes = len(user_preferences)

    def convert_labels(label_list, num_classes):
        label_tensor = torch.zeros(num_classes)
        for label in label_list:
            label_tensor[label] = 1.0
        return label_tensor

    def calculate_class_weights(labels, num_classes):
        label_counts = torch.zeros(num_classes)
        for label in labels:
            label_counts += label

        total_samples = len(labels)
        class_weights = total_samples / (num_classes * label_counts)
        return class_weights

    labels = labels.apply(lambda x: convert_labels(x, num_classes))

    train_data, val_data, train_labels, val_labels = train_test_split(summarized_text, labels, test_size=0.2, random_state=42)

    learning_rate = 1e-6
    batch_size = 4
    num_epochs = 3
    dropout_rate = 0.3
    max_seq_length = 512

    train_dataset = ArticleDataset(train_data, tokenizer, train_labels)
    val_dataset = ArticleDataset(val_data, tokenizer, val_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = ArticleClassifier(num_classes, dropout_rate=dropout_rate).to(device)
    optimizer = AdamW(model.parameters(), learning_rate)

    class_weights = calculate_class_weights(train_labels, num_classes).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)

    for epoch in range(1):
        model.train()
        total_train_loss = 0

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}")

        model.eval()
        total_val_loss = 0
        all_targets = []
        all_preds = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                targets = batch['labels'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs, targets)
                total_val_loss += loss.item()

                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).int()

                all_targets.extend(targets.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='samples')
        accuracy = accuracy_score(all_targets, all_preds)

        print(f"Epoch {epoch + 1}, Val Loss: {avg_val_loss:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, Accuracy: {accuracy:.4f}")

    torch.save(model.state_dict(), model_save_path)
    return model