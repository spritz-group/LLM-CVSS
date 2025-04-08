import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sentence_transformers import SentenceTransformer
import ollama
import random
import sys
import joblib
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"

def setSeed(seed=151836):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

setSeed()

# ========================
# Configuration
# ========================
EMBEDDERS = {
    'all-MiniLM-L6-v2': SentenceTransformer('all-MiniLM-L6-v2'),
    'ATTACK-BERT': SentenceTransformer('basel/ATTACK-BERT'),
    'nomic': None  # handled via ollama
}

CVSS_COMPONENTS = ["AV", "AC", "PR", "UI", "S", "C", "I", "A"]

CLASSIFIERS = {
    'random_forest': lambda: RandomForestClassifier(n_estimators=100),
    'logistic_regression': lambda: LogisticRegression(max_iter=1000),
    'xgboost': lambda: XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
}

TORCH_MODELS = {
    'simple_mlp': lambda in_dim, out_dim: SimpleMLP(in_dim, out_dim),
    'deep_mlp': lambda in_dim, out_dim: DeepMLP(in_dim, out_dim)
}

# ========================
# Dataset Loading
# ========================
df = pd.read_parquet("./dataset.parquet")

# ========================
# Embedding Functions
# ========================
def get_text_input(row, enhanced=False):
    base = [str(row['cve_description']), str(row['assigner'])]
    if enhanced:
        base.extend([str(row['cwe_id']), str(row['cwe_enhanced_description']), str(row['cwe_description']), str(row['cwe_consequences']), str(row['cwe_mitigations'])])
    return " ".join(base)

def embed_sbert(texts, model):
    return model.encode(texts, show_progress_bar=True, batch_size=32)

def embed_nomic(texts):
    return [ollama.embeddings(model="nomic-embed-text", prompt=txt)["embedding"] for txt in tqdm(texts)]

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

# ========================
# Training Loop
# ========================
def train_torch_model(X, y_encoded, model_fn):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim, output_dim = X.shape[1], len(np.unique(y_encoded))
    model = model_fn(input_dim, output_dim).to(device)
    
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_encoded, dtype=torch.long).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for _ in range(30):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        model.eval()
        outputs = model(X_tensor)
        _, predicted = torch.max(outputs, 1)
        acc = (predicted == y_tensor).float().mean()

    cm = confusion_matrix(y_tensor.cpu(), predicted.cpu())
    return model, acc.item(), cm

# ========================
# Main Logic
# ========================
for enhanced in [False, True]:
    setting = "enhanced" if enhanced else "vanilla"
    os.makedirs(f"results/embeddings/{setting}/confusion_matrices", exist_ok=True)
    os.makedirs(f"results/embeddings/{setting}/embeddings", exist_ok=True)
    os.makedirs(f"models/embeddings/{setting}", exist_ok=True)

    for embedder_name, embedder in EMBEDDERS.items():
        print(f"\n[{setting.upper()}] Generating embeddings using {embedder_name}...")

        emb_path = f"results/{setting}/embeddings/{embedder_name}.npy"
        if os.path.exists(emb_path):
            X = np.load(emb_path)
            print(f"Loaded cached embeddings from {emb_path}")
        else:
            print("Generating text inputs...")
            texts = [get_text_input(row, enhanced=enhanced) for _, row in tqdm(df.iterrows(), total=len(df))]

            if embedder_name == "nomic":
                X = np.array(embed_nomic(texts))
            else:
                X = np.array(embed_sbert(texts, embedder))

            np.save(emb_path, X)
            print(f"Saved embeddings to {emb_path}")

        label_encoder = LabelEncoder()
        accuracy_log = {}

        for component in tqdm(CVSS_COMPONENTS, desc=f"[{setting}] Training on CVSS components with {embedder_name}"):
            tqdm.write(f"Training models for {component}")

            y = df[component]
            y_encoded = label_encoder.fit_transform(y)

            # Classical models
            for clf_name, clf_builder in tqdm(CLASSIFIERS.items(), desc="Classical Models"):
                clf = clf_builder()
                X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, stratify=y_encoded, test_size=0.2)
                clf.fit(X_train, y_train)
                preds = clf.predict(X_test)
                acc = accuracy_score(y_test, preds)
                cm = confusion_matrix(y_test, preds)

                joblib.dump(clf, f"models/{setting}/{embedder_name}_{clf_name}_{component}.joblib")
                np.save(f"results/{setting}/confusion_matrices/{embedder_name}_{clf_name}_{component}.npy", cm)
                np.savetxt(f"results/{setting}/confusion_matrices/{embedder_name}_{clf_name}_{component}.txt", cm, fmt="%d")
                tqdm.write(f"{clf_name} Accuracy for {component}: {acc:.4f}")
                accuracy_log.setdefault(clf_name, {})[component] = acc

            # Torch models
            for model_name, model_fn in tqdm(TORCH_MODELS.items(), desc="Torch Models"):
                model, acc, cm = train_torch_model(X, y_encoded, model_fn)
                torch.save(model.state_dict(), f"models/{setting}/{embedder_name}_{model_name}_{component}.pt")
                np.save(f"results/{setting}/confusion_matrices/{embedder_name}_{model_name}_{component}.npy", cm)
                np.savetxt(f"results/{setting}/confusion_matrices/{embedder_name}_{model_name}_{component}.txt", cm, fmt="%d")
                tqdm.write(f"{model_name} Accuracy for {component}: {acc:.4f}")
                accuracy_log.setdefault(model_name, {})[component] = acc

        acc_df = pd.DataFrame.from_dict(accuracy_log, orient="index")
        acc_df.to_csv(f"results/{setting}/{embedder_name}.csv")
        print(f"Saved accuracy log to results/{setting}/{embedder_name}.csv")