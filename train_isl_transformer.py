import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from model import TransformerModel
from dataset import ASLDataset, ASLTestDataset, ISLDataset, ISLTestDataset
import os
import torch.nn.functional as F
import data_augment as augment
import create_variation_df_asl as cvda
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
import time
from sklearn.metrics import classification_report
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# va452@ogg.cs.bath.ac.uk:/mnt/fast0/va452
# rsync /Users/vrushaliashok/sign_language/model.py va452@ogg.cs.bath.ac.uk:/mnt/fast0/va452
# rsync -r va452@ogg.cs.bath.ac.uk:/mnt/fast0/va452/CL_t-SNE /Users/vrushaliashok/sign_language/topk_accuracies
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
batch_size = 128

isl_train_list = np.load('ISL_WORDS.npy')
full_train_isl_df = pd.read_pickle("ISL_dataset_train_FULL.pkl")
test_isl_df = pd.read_pickle("ISL_dataset_test_FULL.pkl")

class_counts = full_train_isl_df['ISL_GLOSSES'].value_counts()
valid_classes = ['baby', 'bad', 'beautiful', 'bicycle', 'black', 'book', 'boy', 'brother', 'car', 'cat', 'child', 'cold', 'court', 'deaf', 'deep', 'doctor', 'dog', 'expensive', 'family', 'famous', 'fast', 'father', 'fish', 'friend', 'gift', 'happy', 'hard', 'heavy', 'high', 'horse', 'hospital', 'hot', 'house', 'husband', 'key', 'king', 'laptop', 'library', 'loud', 'man', 'mean', 'minute', 'money', 'morning', 'mother', 'new', 'night', 'old', 'paper', 'president', 'quiet', 'red', 'school', 'sick', 'sign', 'sister', 'slow', 'son', 'student', 'table', 'time', 'tomorrow', 'warm', 'wide', 'wife', 'window', 'woman', 'year', 'yesterday']

# valid_classes = ['baby', 'bad', 'beautiful', 'bicycle', 'book', 'boy', 'brother', 'car', 'cat', 'child', 'cold', 'court', 'deaf', 'dog', 'expensive', 'family', 'fast', 'father', 'friend', 'gift', 'happy', 'hard', 'high', 'horse', 'hospital', 'hot', 'house', 'husband', 'key', 'king', 'laptop', 'library', 'loud', 'man', 'mean', 'money', 'mother', 'new', 'night', 'old', 'paper', 'president', 'quiet', 'school', 'sick', 'sign', 'sister', 'student', 'time', 'tomorrow', 'warm', 'wife', 'window', 'woman', 'year', 'yesterday']

full_train_isl_df = full_train_isl_df[full_train_isl_df['ISL_GLOSSES'].isin(valid_classes)]
class_counts = full_train_isl_df['ISL_GLOSSES'].value_counts()
test_isl_df = test_isl_df[test_isl_df['ISL_GLOSSES'].isin(valid_classes)]
print(f"Classes remaining: {full_train_isl_df['ISL_GLOSSES'].nunique()}")
print(f"Samples remaining: {len(full_train_isl_df)}")

unique_glosses = sorted(full_train_isl_df['ISL_GLOSSES'].unique())
gloss_to_idx = {g: i for i, g in enumerate(unique_glosses)}

model_type = 'T'
train_isl_df = full_train_isl_df
print(len(np.unique(train_isl_df['ISL_GLOSSES'])), len(np.unique(test_isl_df['ISL_GLOSSES'])))
train_dataset = ISLTestDataset(train_isl_df, gloss_to_idx, model_type)
test_dataset = ISLTestDataset(test_isl_df, gloss_to_idx, model_type)

print('len(train_dataset)', len(train_dataset))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print('len(train_loader)', len(train_loader), len(train_loader)*batch_size)

sample_batch, label_batch = next(iter(train_loader))
B, T, D = sample_batch.shape

print(B,T,D)
num_classes = full_train_isl_df['ISL_GLOSSES'].nunique()
print('num_classes', num_classes)

def calc_accuracy(model, loader, device):
    model.eval()
    correct_top1 = 0
    correct_top3 = 0
    correct_top5 = 0
    correct_top10 = 0

    total = 0
    all_preds = []
    all_y = []
    features = []
    labels = []
    test_loss = 0
    for x, y in loader:
        x = x.to(device).float()
        y = y.to(device)
        enc, fts, logits = model(x)
        features.append(fts)
        labels.append(y)
        predictions = logits.argmax(dim=1)
        loss = criterion(logits, y)
        test_loss += loss.item()
        # Top-1
        predictions = logits.argmax(dim=1)
        correct_top1 += (predictions == y).sum().item()
        
        # Top-3
        topk_preds = torch.topk(logits, k=3, dim=1).indices  # (batch, k)
        correct_top3 += (topk_preds == y.unsqueeze(1)).any(dim=1).sum().item()

         # Top-5
        topk_preds = torch.topk(logits, k=5, dim=1).indices  # (batch, k)
        correct_top5 += (topk_preds == y.unsqueeze(1)).any(dim=1).sum().item()

         # Top-10
        topk_preds = torch.topk(logits, k=10, dim=1).indices  # (batch, k)
        correct_top10 += (topk_preds == y.unsqueeze(1)).any(dim=1).sum().item()
        
        total += y.numel()
        all_preds.append(predictions.cpu())
        all_y.append(y.cpu())
    features = torch.cat(features, dim=0).cpu().detach().numpy()
    labels = torch.cat(labels, dim=0).cpu().numpy()

    top1 = (correct_top1 / total) * 100
    top3 = (correct_top3 / total) * 100
    top5 = (correct_top5 / total) * 100
    top10 = (correct_top10 / total) * 100
    all_preds, all_y = torch.cat(all_preds), torch.cat(all_y)
    
    report = classification_report(
        all_y.numpy(),
        all_preds.numpy(),
        target_names=unique_glosses,
        zero_division=0, 
        output_dict=True
    )

    return top1, top3, top5, top10, all_preds, all_y, test_loss, report, features, labels


model = TransformerModel(input_dim=D, ntoken = num_classes, d_model = 128, nhead=2, d_hid=256, nlayers= 2, dropout = 0.1, batch_first=False, norm_first=True).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=1e-3)
# optimizer = AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False, maximize=False, foreach=None, capturable=False, differentiable=False, fused=None)

epochs = 500
experiment_no = 'ISL-HT-10'
component = f'Best model'

complete_losses = []
test_losses = []
top1_accs = []
top3_accs = []
top5_accs = []
top10_accs = []
val_accuracies = []
features = []
labels = []
start = time.time()
for epoch in range(epochs):
    print('Start of epoch ', epoch+1)
    model.train()
    total_loss = 0
    features_tr = []
    labels_tr = []
    for x, y in train_loader:
        x = x.to(device).float()   # (B, T, D)
        y = y.to(device)           # (B,)
        optimizer.zero_grad()
        enc_output, fts, outputs = model(x)
        features_tr.append(fts)
        labels_tr.append(y)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    features_tr = torch.cat(features_tr, dim=0).cpu().detach().numpy()
    labels_tr = torch.cat(labels_tr, dim=0).cpu().numpy()
    complete_losses.append(total_loss)
    print("Training done")
    torch.save(model.state_dict(), f'isl_transformer_{experiment_no}.pt')
    
    top1, top3, top5, top10, predictions, y, test_loss, report, features, labels = calc_accuracy(model, test_loader, device)
    test_losses.append(test_loss)
    top1_accs.append(top1)
    top3_accs.append(top3)
    top5_accs.append(top5)
    top10_accs.append(top10)
    
    if(epoch+1) == 500:
        # print(report)
         # Convert to dataframe, drop summary rows
        report_df = pd.DataFrame(report).T
        report_df = report_df.drop(index=['accuracy', 'macro avg', 'weighted avg'])
        report_df = report_df.sort_values('f1-score', ascending=False)
        
        top5_cl    = list(report_df.head(5).index)
        bottom5_cl = list(report_df.tail(5).index)
        
        print('Best 5:', top5_cl)
        print('Worst 5:', bottom5_cl)
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
    if (epoch+1)%10 == 0:
        print("Top 1 Accuracy:", top1,"%")
        print("Top 3 Accuracy:", top3,"%")
        print("Top 5 Accuracy:", top5,"%")
        print("Top 10 Accuracy:", top10,"%")



fig, ax = plt.subplots()
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
tsne_results = tsne.fit_transform(features_tr)

sil = silhouette_score(tsne_results, labels_tr)
db  = davies_bouldin_score(tsne_results, labels_tr)
ch  = calinski_harabasz_score(tsne_results, labels_tr)

scatter = plt.scatter(
    tsne_results[:, 0],
    tsne_results[:, 1],
    c=labels_tr,
    cmap='tab20',
    alpha=0.7
)

fig.colorbar(scatter)
ax.set_xlabel("Dim 1")
ax.set_ylabel("Dim 2")

ax.set_title(
    f"{component} Training\nSil: {sil:.3f}, DB: {db:.3f}, CH: {ch:.1f}",
    fontsize=7,
    pad=2
)
fig.savefig(f"ISL_Graphs_Final/Exp. {experiment_no}, Training TSNE.png")

ax.clear()
fig3, ax3 = plt.subplots()
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
tsne_results = tsne.fit_transform(features)

sil = silhouette_score(tsne_results, labels)
db  = davies_bouldin_score(tsne_results, labels)
ch  = calinski_harabasz_score(tsne_results, labels)

scatter = plt.scatter(
    tsne_results[:, 0],
    tsne_results[:, 1],
    c=labels,
    cmap='tab20',
    alpha=0.7
)

fig3.colorbar(scatter)
ax3.set_xlabel("Dim 1")
ax.set_ylabel("Dim 2")

ax3.set_title(
    f"{component}\nSil: {sil:.3f}, DB: {db:.3f}, CH: {ch:.1f}",
    fontsize=7,
    pad=2
)
fig3.savefig(f"ISL_Graphs_Final/Exp. {experiment_no}, Testing TSNE.png")

ax3.clear()

fig2, ax2 = plt.subplots()
# Loss plot
ax2.plot(range(epochs), complete_losses, c='mediumvioletred')
ax2.set_title(f"{component}, Training Loss")
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Training Loss")
fig2.savefig(f"ISL_Graphs_Final/Exp. {experiment_no}, Training Loss.png")


ax2.clear()
# Loss plot
ax2.plot(range(epochs), test_losses, c='crimson')
ax2.set_title(f"{component}, Testing Loss")
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Testing Loss")
fig2.savefig(f"ISL_Graphs_Final/Exp. {experiment_no}, Testing Loss.png")
# Clear axes (not figure)
ax2.clear()

# Accuracy plot
ax2.plot(range(epochs), top1_accs, c="#675dbcff", label = 'Top-1')
ax2.plot(range(epochs), top3_accs, c="#62c319ff", label = 'Top-3')
ax2.plot(range(epochs), top5_accs, c="#c93763", label = 'Top-5')
ax2.plot(range(epochs), top10_accs, c="#459ed1ff", label = 'Top-10')
ax2.set_title(f"{component}, Top-k Accuracies")
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Accuracy")
ax2.legend()
fig2.savefig(f"ISL_Graphs_Final/Exp. {experiment_no}, Accuracy.png")

ax2.clear()
mins = (time.time() - start) / 60
print(experiment_no)
print(f"Run finished in {mins:.2f} mins")

