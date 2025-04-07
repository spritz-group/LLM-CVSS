import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sentence_transformers import SentenceTransformer
import ollama
import joblib
from tqdm import tqdm

# ========================
# Configuration
# ========================
EMBEDDERS = {
    'all-MiniLM-L6-v2': SentenceTransformer('all-MiniLM-L6-v2'),
    'ATTACK-BERT': SentenceTransformer('basel/ATTACK-BERT'),
    'nomic': None  # placeholder, used in custom embedding function
}

CVSS_COMPONENTS = ["AV", "AC", "PR", "UI", "S", "C", "I", "A"]

CLASSIFIERS = {
    'random_forest': lambda: RandomForestClassifier(n_estimators=100),
    'logistic_regression': lambda: LogisticRegression(max_iter=1000),
    'xgboost': lambda: XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
}

# ========================
# Dataset Loading
# ========================
df = pd.read_parquet("./dataset.parquet")

# ========================
# Embedding Functions
# ========================
def get_text_input(row):
    return " ".join([
        str(row['cve_description']), str(row['cwe_description']),
        str(row['cwe_id']), str(row['assigner']),
        str(row['cwe_enhanced_description']), str(row['cwe_consequences']),
        str(row['cwe_mitigations'])
    ])

def embed_sbert(row, model):
    return model.encode(get_text_input(row))

def embed_nomic(row):
    response = ollama.embeddings(model="nomic-embed-text", prompt=get_text_input(row))
    return response["embedding"]

# ========================
# PyTorch Models
# ========================
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.model(x)

class DeepMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.model(x)

TORCH_MODELS = {
    'simple_mlp': SimpleMLP,
    'deep_mlp': DeepMLP
}

# ========================
# Training Loop
# ========================

def train_torch_model(X, y_encoded, torch_model, component):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    input_dim = X.shape[1]
    output_dim = len(np.unique(y_encoded))
    
    model = torch_model(input_dim=input_dim, output_dim=output_dim).to(device)
    
    X_train_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_encoded, dtype=torch.long).to(device)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(30):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        model.eval()
        outputs = model(X_train_tensor)
        _, predicted = torch.max(outputs, 1)
        acc = (predicted == y_train_tensor).float().mean()

    return model, acc.item()


# ========================
# Create folders
# ========================
os.makedirs("results/embeddings", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ========================
# Main Execution
# ========================
for embedder_name, embedder in EMBEDDERS.items():
    print(f"\nGenerating embeddings using {embedder_name}...")

    embedding_path = f"results/embeddings/{embedder_name}.npy"
    if os.path.exists(embedding_path):
        X = np.load(embedding_path)
        print(f"Loaded cached embeddings from {embedding_path}")
    else:
        if embedder_name == 'nomic':
            embeddings = []
            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Embedding with {embedder_name}"):
                embeddings.append(embed_nomic(row))
        else:
            print("Generating text inputs for batch embedding...")
            texts = [get_text_input(row) for _, row in tqdm(df.iterrows(), total=len(df))]
            print("Encoding in batches...")
            embeddings = embedder.encode(texts, show_progress_bar=True, batch_size=32)
        
        X = np.array(embeddings)
        np.save(embedding_path, X)
        print(f"Saved embeddings to {embedding_path}")

    label_encoder = LabelEncoder()
    
    accuracy_log = {}

    for component in tqdm(CVSS_COMPONENTS, desc=f"Training on CVSS components ({embedder_name})"):
        tqdm.write(f"Training models for {component} using {embedder_name} embeddings...")
        
        y = df[component]
        y_encoded = label_encoder.fit_transform(y)
        
        # Classic classifiers
        for clf_name, clf_builder in tqdm(CLASSIFIERS.items(), desc="ML Classifiers"):
            tqdm.write(f"Training {clf_name} for {component}")
            clf = clf_builder()
            X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, stratify=y_encoded, test_size=0.2)
            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)
            acc = accuracy_score(y_test, preds)
            tqdm.write(f"{clf_name} {component} Accuracy: {acc:.4f}")
            joblib.dump(clf, f"models/{embedder_name}_{clf_name}_{component}.joblib")
            accuracy_log.setdefault(clf_name, {})[component] = acc

        # Torch models
        for torch_name, torch_model in tqdm(TORCH_MODELS.items(), desc="Torch Models"):
            tqdm.write(f"Training {torch_name} for {component}")
            model, acc = train_torch_model(X, y_encoded, torch_model, component)  # Use encoded labels here
            torch.save(model.state_dict(), f"models/{embedder_name}_{torch_name}_{component}.pt")
            tqdm.write(f"Torch {torch_name} {component} Accuracy: {acc:.4f}")
            accuracy_log.setdefault(torch_name, {})[component] = acc

    acc_df = pd.DataFrame.from_dict(accuracy_log, orient='index')
    acc_df.to_csv(f"results/{embedder_name}.csv")
    print(f"Saved accuracy results to results/{embedder_name}.csv")