import pandas as pd
import numpy as np
import chromadb
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import balanced_accuracy_score, f1_score, matthews_corrcoef, confusion_matrix
from utils import record_to_enhanced_explicit_text, CATEGORICAL_MAPPINGS
import torch

# --- CONFIGURATION ---
MODEL_NAME = "BAAI/bge-large-en-v1.5"
DB_PATH = "./local_chroma_db"
PREDICTIONS_FILE = os.path.join("data", "predictions.csv")
TEST_DATA_FILE = os.path.join("data", "X_test.csv")
TEST_LABELS_FILE = os.path.join("data", "y_test.csv")
SAMPLE_SIZE = 10000  # Full evaluation

def run_full_evaluation():
    print("FARASAT EVALUATION PIPELINE (TIER 1 & 2)")
    print("="*60)

    # ---------------------------------------------------------
    # TIER 1: END-TO-END PERFORMANCE
    # ---------------------------------------------------------
    print("\n[1/2] CALCULATING TIER 1 METRICS (PREDICTION QUALITY)...")
    try:
        df_pred = pd.read_csv(PREDICTIONS_FILE)
        y_true = df_pred['actual']
        y_pred = df_pred['predicted']
        
        # 1. Metrics
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        f1_long = f1_score(y_true, y_pred, pos_label=1)
        f1_short = f1_score(y_true, y_pred, pos_label=0)
        f1_macro = f1_score(y_true, y_pred, average='macro')
        mcc = matthews_corrcoef(y_true, y_pred)
        
        # 2. Confusion Matrix Calculation
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        print(f"   Balanced Accuracy: {bal_acc:.4f}")
        print(f"   F1-Score (Long):   {f1_long:.4f}")
        print(f"   F1-Score (Short):  {f1_short:.4f}")
        print(f"   F1-Score (Macro):  {f1_macro:.4f}")
        print(f"   MCC:               {mcc:.4f}")
        print(f"   Counts: TP={tp} | TN={tn} | FP={fp} | FN={fn}")

        # 3. Draw & Save Confusion Matrix
        print("   Generating Confusion Matrix Image...")
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Predicted Short', 'Predicted Long'],
                    yticklabels=['Actual Short', 'Actual Long'])
        plt.title(f"Farasat Confusion Matrix (N={len(y_true)})")
        plt.ylabel("Ground Truth")
        plt.xlabel("Model Prediction")
        
        # Save file
        plt.savefig("confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("   Saved to 'confusion_matrix.png'")
        
    except Exception as e:
        print(f"   Tier 1 Failed: {e}")

    # ---------------------------------------------------------
    # TIER 2: RETRIEVAL QUALITY
    # ---------------------------------------------------------
    print("\n[2/2] CALCULATING TIER 2 METRICS (RAG QUALITY)...")
    try:
        if not os.path.exists(TEST_DATA_FILE): raise FileNotFoundError(f"Missing {TEST_DATA_FILE}")
             
        df_test = pd.read_csv(TEST_DATA_FILE)
        y_test_full = pd.read_csv(TEST_LABELS_FILE)["TARGET"]
        
        if SAMPLE_SIZE:
            df_test = df_test.iloc[:SAMPLE_SIZE]
            y_test_full = y_test_full.iloc[:SAMPLE_SIZE]
            print(f" Running on {SAMPLE_SIZE} random samples.")

        model = SentenceTransformer(MODEL_NAME)
        client = chromadb.PersistentClient(path=DB_PATH)
        collection = client.get_collection("production_db")
        
        concordance_scores = []
        similarity_scores = []
        
        print("   ... Querying Vector Database ...")
        for i, row in tqdm(df_test.iterrows(), total=len(df_test)):
            text = record_to_enhanced_explicit_text(row.to_dict(), CATEGORICAL_MAPPINGS)
            
            # 1. Embed Query -> Force to Float32
            query_emb = model.encode(text, convert_to_tensor=True).float()
            
            results = collection.query(
                query_embeddings=query_emb.tolist(),
                n_results=5,
                include=['metadatas', 'embeddings']
            )
            
            # 2. Label Concordance
            actual_label = int(y_test_full.iloc[i])
            neighbor_labels = [m['los_label'] for m in results['metadatas'][0]]
            majority_vote = 1 if sum(neighbor_labels) >= 3 else 0
            concordance_scores.append(1 if majority_vote == actual_label else 0)
            
            # 3. Semantic Relevance -> Force to Float32 & Same Device
            retrieved_embs = torch.tensor(results['embeddings'][0], dtype=torch.float32).to(query_emb.device)
            
            cos_scores = util.cos_sim(query_emb, retrieved_embs)[0]
            similarity_scores.append(float(cos_scores.mean()))

        avg_concordance = np.mean(concordance_scores)
        avg_similarity = np.mean(similarity_scores)
        
        print(f"   Label-Concordance@5:  {avg_concordance:.2%}")
        print(f"   Semantic Relevance:   {avg_similarity:.4f}")

    except Exception as e:
        print(f"   Tier 2 Failed: {e}")
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)

if __name__ == "__main__":
    run_full_evaluation()