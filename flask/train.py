from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn.functional as F
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from model import ArticleClassifier
from ArticleDataset import ArticleDataset
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

def load_training_data(user_preferences):
    # TODO
    # DB에서 preference 해당하는 train_data 가져오기 구현 필요
    user_preferences = user_preferences # 리스트 형태라 가정


    article_df = pd.read_excel('sum_article_extend.xlsx')
    personal_article_df  = pd.DataFrame()

    for preference in user_preferences:
        filtered_articles= article_df[article_df['tag'].str.contains(preference,case=False,na=False)]
        personal_article_df = pd.concat([personal_article_df, filtered_articles])

    return personal_article_df

def train_save_personal_classifier(train_data,tokenizer, user_preferences,userID):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_save_path = f'{userID}.pth'
    personal_article_df = train_data
    summarized_text = personal_article_df['summarized']
    labels = personal_article_df['tags_encoded']
    train_data, val_data, train_labels, val_labels = train_test_split(summarized_text,labels, test_size=0.2, random_state=42)

    train_dataset = ArticleDataset(train_data, tokenizer, train_labels)
    val_dataset = ArticleDataset(val_data, tokenizer, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False) 

    model = ArticleClassifier(num_classes=len(user_preferences)).to(device)
    optimizer = Adam(model.parameters(), lr=1e-5)
    patience = 3
    best_val_loss = float('inf')
    patience_counter = 0
    for epoch in range(30):  # 에폭 수를 30으로 설정
        model.train()
        total_loss = 0
        for input_ids, attention_mask, targets in train_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            targets = targets.to(device, dtype=torch.long)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = F.cross_entropy(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss}")

        # 검증 루프
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for input_ids, attention_mask, targets in val_loader:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                targets = targets.to(device, dtype=torch.long)
                outputs = model(input_ids, attention_mask)
                loss = F.cross_entropy(outputs, targets)
                val_loss += loss.item()
                predicted_classes = torch.argmax(F.softmax(outputs, dim=1), dim=1)
                correct += (predicted_classes == targets).sum().item()
                total += targets.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = correct / total
        print(f"Validation Loss: {avg_val_loss}, Validation Accuracy: {val_accuracy:.2f}")

        # 조기 종료 체크
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
    torch.save(model.state_dict(),model_save_path)
    return model