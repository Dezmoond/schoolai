import torch
import pandas as pd
import pickle
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from tqdm import tqdm
import re
import pandas as pd
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader

# Загрузка данных
train_df = pd.read_csv('train.csv')  # Обратите внимание на правильный путь к файлу
test_df = pd.read_csv('test.csv')

# Проверим, как выглядят данные
print(train_df.head())
print(test_df.head())


# Определим функцию предобработки текста
def preprocess_text(text):
    text = text.lower()  # Пример обработки: приводим к нижнему регистру
    # Здесь можно добавить другие шаги предобработки, например, удаление стоп-слов, пунктуации и т.д.
    return text


# Применяем предобработку
train_df['text'] = train_df['text'].apply(preprocess_text)
test_df['text'] = test_df['text'].apply(preprocess_text)

# Загружаем токенизатор
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')  # Пример токенизатора

# Преобразуем метки уровней в числовые значения
level_mapping = {'A1': 0, 'A2': 1, 'B1': 2, 'B2': 3, 'C1': 4, 'C2': 5}
train_df['label'] = train_df['level'].map(level_mapping)


# Создаем Dataset класс для текста
class CEFRDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        text = self.dataframe.iloc[idx]['text']
        label = self.dataframe.iloc[idx]['label']

        # Токенизация текста
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        # Возвращаем тензоры с текстовыми признаками и меткой
        return {
            'input_ids': encoding['input_ids'].squeeze(),  # Сжимаем размерность
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# Создаем Dataset для обучающей и тестовой выборки
train_dataset = CEFRDataset(train_df, tokenizer)
test_dataset = CEFRDataset(test_df, tokenizer)

# Создаем DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Пример использования:
for batch in train_dataloader:
    print(batch)
    break  # Для проверки вывода первого батча

# Функция предобработки текста
def preprocess_text(text):
    text = re.sub(r"([.,!?;()\-])", r" \1 ", text)
    text = re.sub(r"[^a-zA-Z.,!?;()\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


# Класс Dataset
class CEFRDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        self.labels = torch.tensor(labels)

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}, self.labels[idx]

    def __len__(self):
        return len(self.labels)


# Функция сохранения модели и метаданных
def save_model(model, tokenizer, label_encoder, model_path="cefr_model.pth"):
    torch.save(model.state_dict(), model_path)
    tokenizer.save_pretrained("tokenizer")
    with open("label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)
    print(f"Model and tokenizer saved to {model_path}")


# Функция загрузки модели и метаданных
def load_model(model_path="cefr_model.pth", device="cpu"):
    # Убедимся, что модель загружается на правильное устройство
    model = AutoModelForSequenceClassification.from_pretrained(
        "xlm-roberta-base",
        num_labels=6
    ).to(device)

    # Загружаем веса с указанием устройства
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Загружаем токенизатор и label_encoder
    tokenizer = AutoTokenizer.from_pretrained("tokenizer")
    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    print("Model loaded successfully.")
    return model, tokenizer, label_encoder


# Функция обучения модели
def train_model(model, train_loader, tokenizer, label_encoder, val_loader, optimizer, scheduler, device, epochs=5):
    best_f1 = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        all_preds, all_labels = [], []
        for batch, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            optimizer.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = labels.to(device)
            outputs = model(**batch, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            all_preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        val_acc, val_f1 = evaluate(model, val_loader, device)
        print(
            f"Epoch {epoch + 1}: Train Loss: {total_loss / len(train_loader):.4f}, Val Accuracy: {val_acc:.4f}, Val F1: {val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            save_model(model, tokenizer, label_encoder, "cefr_model.pth")


# Функция оценки
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch, labels in loader:
            # Переносим все данные на нужное устройство
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = labels.to(device)
            outputs = model(**batch)
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return accuracy_score(all_labels, all_preds), f1_score(all_labels, all_preds, average='macro')


# Основной пайплайн
def main():
    MODEL_NAME = "xlm-roberta-base"  # Используем предобученную модель XLM-RoBERTa
    BATCH_SIZE = 64
    EPOCHS = 15
    LEARNING_RATE = 3e-5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Загрузка и подготовка данных
    df_train = pd.read_csv("train.csv")
    df_train['text'] = df_train['text'].apply(preprocess_text)
    label_encoder = LabelEncoder()
    df_train['level'] = label_encoder.fit_transform(df_train['level'])
    print("Class labels:", label_encoder.classes_)

    # Разделение данных
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df_train['text'].tolist(),
        df_train['level'].tolist(),
        test_size=0.2,
        stratify=df_train['level']
    )

    # Инициализация токенизатора и датасетов
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = CEFRDataset(train_texts, train_labels, tokenizer)
    val_dataset = CEFRDataset(val_texts, val_labels, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Инициализация модели
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_encoder.classes_)
    ).to(device)

    # Оптимизатор и шедулер
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=500,
        num_training_steps=len(train_loader) * EPOCHS
    )

    # Обучение модели
    print("Starting training...")
    train_model(model, train_loader, tokenizer, label_encoder, val_loader, optimizer, scheduler, device, epochs=EPOCHS)

    # Загрузка и тестирование модели
    print("\nLoading best model for testing...")
    model, tokenizer, label_encoder = load_model("cefr_model.pth", device=device)

    # Важно: пересоздаем DataLoader с правильным токенизатором
    val_dataset = CEFRDataset(val_texts, val_labels, tokenizer)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    test_acc, test_f1 = evaluate(model, val_loader, device)
    print(f"\nTest Accuracy: {test_acc:.4f}, Test F1: {test_f1:.4f}")


if __name__ == "__main__":
    main()
