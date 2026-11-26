import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from model import TransformerModel
from dataset import PairedDataset, ISLTestDataset, ASLTestDataset, TestDataset
import pandas as pd
import matplotlib.pyplot as plt
import create_variation_df_asl as cvda
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import os
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colorbar as mcolorbar
import matplotlib.colors as mcolors
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from losses import SupConLoss
import time
from sklearn.metrics import classification_report


# define loss with a temperature `temp`

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from infonce import SupervisedInfoNCE

from sklearn.metrics import classification_report
import pandas as pd
def retrieve_test(isl_model, asl_model, test_dataset, isl_proj, asl_proj, epoch, device, idx2label=None):
    isl_model.eval()
    asl_model.eval()

    isl_data, isl_labels = test_dataset.get_isl()
    asl_data, asl_labels = test_dataset.get_asl()

    isl_data, isl_labels = test_dataset.get_isl()
    asl_data, asl_labels = test_dataset.get_asl()

    # build simple loaders for ISL and ASL separately
    isl_loader = DataLoader(torch.utils.data.TensorDataset(isl_data, isl_labels), batch_size=64, shuffle=False)
    asl_loader = DataLoader(torch.utils.data.TensorDataset(asl_data, asl_labels), batch_size=64, shuffle=False)

    all_z_i, all_z_j = [], []

    with torch.no_grad():
        for x_i, _ in isl_loader:
            x_i = x_i.to(device)
            z_i, _, _ = isl_model(x_i)
            z_i = z_i.mean(dim=1)
            z_i = isl_proj(z_i)
            z_i = F.normalize(z_i, p=2, dim=-1)
            all_z_i.append(z_i.cpu())

        for x_j, _ in asl_loader:
            x_j = x_j.to(device)
            z_j, _, _ = asl_model(x_j)
            z_j = z_j.mean(dim=1)
            z_j = asl_proj(z_j)
            z_j = F.normalize(z_j, p=2, dim=-1)
            all_z_j.append(z_j.cpu())

    all_z_i = F.normalize(torch.cat(all_z_i), dim=1)
    all_z_j = F.normalize(torch.cat(all_z_j), dim=1)

    sim_matrix = torch.matmul(all_z_i, all_z_i.T)  # (N, N)

    # mean similarity between same class
    # mean similarity between different classes
    labels = isl_labels
    same_mask = (labels.unsqueeze(0) == labels.unsqueeze(1))
    diff_mask = ~same_mask
    same_mask.fill_diagonal_(False)
    intra_cs = sim_matrix[same_mask].mean().item()
    inter_cs = sim_matrix[diff_mask].mean().item()

    print("mean intra-class similarity:", sim_matrix[same_mask].mean().item())
    print("mean inter-class similarity:", sim_matrix[diff_mask].mean().item())
   

    def topk_accuracy(queries, query_labels, gallery, gallery_labels, k):
        similarity = torch.matmul(queries, gallery.T)
        topk_indices = torch.topk(similarity, k=k, dim=1).indices
        retrieved_labels = gallery_labels[topk_indices]
        correct = (retrieved_labels == query_labels.unsqueeze(1)).any(dim=1)
        return correct

    def aggregate(correct, labels):
        overall = correct.float().mean().item() * 100
        per_class = {}
        for cls in labels.unique():
            mask = labels == cls
            per_class[cls.item()] = correct[mask].float().mean().item() * 100
        return overall, per_class

    def get_top1_preds(queries, gallery, gallery_labels):
        similarity = torch.matmul(queries, gallery.T)
        top1_indices = similarity.argmax(dim=1)
        return gallery_labels[top1_indices]

    def print_classification_report(query_labels, pred_labels, direction, idx2label=None):
        y_true = query_labels.numpy()
        y_pred = pred_labels.numpy()
        target_names = [idx2label[i] for i in sorted(set(y_true))] if idx2label else None
        print(f"\n  Classification Report ({direction}):")
        print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))

    results = {}
    per_class_results = {}

    for k in [1, 3, 5, 10]:
        ia_correct = topk_accuracy(all_z_i, isl_labels, all_z_j, asl_labels, k)
        ai_correct = topk_accuracy(all_z_j, asl_labels, all_z_i, isl_labels, k)
        results[f'IA_top{k}'], per_class_results[f'IA_top{k}'] = aggregate(ia_correct, isl_labels)
        results[f'AI_top{k}'], per_class_results[f'AI_top{k}'] = aggregate(ai_correct, asl_labels)

    # Top-1 predictions for classification report
    ia_preds = get_top1_preds(all_z_i, all_z_j, asl_labels)
    ai_preds = get_top1_preds(all_z_j, all_z_i, isl_labels)

    # Overall
    if (epoch+1)%100 == 0:
        print(f"  ISL-ASL  Top-1: {results['IA_top1']:.2f}%  Top-3: {results['IA_top3']:.2f}%  Top-5: {results['IA_top5']:.2f}%  Top-10: {results['IA_top10']:.2f}%")
        print(f"  ASL-ISL  Top-1: {results['AI_top1']:.2f}%  Top-3: {results['AI_top3']:.2f}%  Top-5: {results['AI_top5']:.2f}%  Top-10: {results['AI_top10']:.2f}%")
    IA_top_1, IA_top_3, IA_top_5, IA_top_10 = results['IA_top1'], results['IA_top3'], results['IA_top5'], results['IA_top10']
    AI_top_1, AI_top_3, AI_top_5, AI_top_10 = results['AI_top1'], results['AI_top3'], results['AI_top5'], results['AI_top10']
    # items = sorted(per_class_results['IA_top1'].items(), key=lambda x: x[1], reverse=True)


    return results, intra_cs, inter_cs, [IA_top_1, IA_top_3], [AI_top_1, AI_top_3]

def plot_tsne(isl_model, asl_model, loader, device, gloss_to_idx, epoch, experiment_no, title="t-SNE of ISL vs ASL Embeddings"):
    isl_model.eval()
    asl_model.eval()

    all_z_i, all_z_j, all_glosses = [], [], []

    with torch.no_grad():
        for x_i, x_j, gloss in loader:
            x_i, x_j = x_i.to(device), x_j.to(device)

            z_i, _, _ = isl_model(x_i)
            z_j, _, _ = asl_model(x_j)
            z_i = z_i.mean(dim=1)
            z_j = z_j.mean(dim=1)

            z_i = F.normalize(z_i, p=2, dim=-1)
            z_j = F.normalize(z_j, p=2, dim=-1)

            all_z_i.append(z_i.cpu().numpy())
            all_z_j.append(z_j.cpu().numpy())
            all_glosses.extend(gloss)

    all_z_i = np.concatenate(all_z_i, axis=0)
    all_z_j = np.concatenate(all_z_j, axis=0)

    combined = np.concatenate([all_z_i, all_z_j], axis=0)
    labels   = all_glosses + all_glosses
    source   = ["ISL"] * len(all_z_i) + ["ASL"] * len(all_z_j)

    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init="pca", learning_rate="auto")
    reduced = tsne.fit_transform(combined)

    # Map glosses to integer indices for coloring
    unique_glosses = sorted(set(labels))
    gloss_to_int   = {g: i for i, g in enumerate(unique_glosses)}
    color_indices  = np.array([gloss_to_int[l] for l in labels], dtype=float)

    n = len(unique_glosses)
    cmap = plt.colormaps["tab20"].resampled(n)
    norm  = mcolors.BoundaryNorm(boundaries=np.arange(-0.5, n, 1), ncolors=n)

    fig, ax = plt.subplots(figsize=(12, 8))

    for i, (x, y) in enumerate(reduced):
        marker = "o" if source[i] == "ISL" else "^"
        ax.scatter(x, y, c=[color_indices[i]], cmap=cmap, norm=norm,
                   marker=marker, s=60, alpha=0.7, edgecolors="white", linewidths=0.3)

    # Colorbar for glosses
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_ticks([])
    cbar.set_label("Gloss", fontsize=11)

    # Modality legend (shape only)
    modality_handles = [
        plt.Line2D([0], [0], marker="o", color="grey", markersize=8, linestyle="None", label="ISL"),
        plt.Line2D([0], [0], marker="^", color="grey", markersize=8, linestyle="None", label="ASL"),
    ]
    ax.legend(handles=modality_handles, title="Modality", loc="upper left", fontsize=9)

    ax.set_title(title)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    plt.tight_layout()
    plt.savefig(f"CL_Graphs_Final/{experiment_no}_tsne_{epoch+1}.png", dpi=150, bbox_inches="tight")
    plt.show()

def cross_modal_euclidean(z_i, z_j, labels):
    """
    Mean euclidean distance between ISL and ASL centroids for each class.
    Lower = better aligned.
    z_i, z_j: (N, D) numpy arrays
    labels: (N,) numpy array
    """
    unique_labels = np.unique(labels)
    dists = []
    for l in unique_labels:
        mask = labels == l
        if mask.sum() < 1:
            continue
        c_i = z_i[mask].mean(axis=0)  # ISL centroid for class l
        c_j = z_j[mask].mean(axis=0)  # ASL centroid for class l
        dist = np.linalg.norm(c_i - c_j)  # euclidean distance
        dists.append(dist)
    
    return np.mean(dists), dict(zip(unique_labels, dists))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
shared_glosses = ['baby', 'bad', 'beautiful', 'bicycle', 'black', 'book', 'boy', 'brother', 'car', 'cat', 'child', 'cold', 'court', 'deaf', 'deep', 'doctor', 'dog', 'expensive', 'family', 'famous', 'fast', 'father', 'fish', 'friend', 'gift', 'happy', 'hard', 'heavy', 'high', 'horse', 'hospital', 'hot', 'house', 'husband', 'key', 'king', 'laptop', 'library', 'loud', 'man', 'mean', 'minute', 'money', 'morning', 'mother', 'new', 'night', 'old', 'paper', 'president', 'quiet', 'red', 'school', 'sick', 'sign', 'sister', 'slow', 'son', 'student', 'table', 'time', 'tomorrow', 'warm', 'wide', 'wife', 'window', 'woman', 'year', 'yesterday']
# shared_glosses = ['baby', 'bad', 'beautiful', 'bicycle', 'book', 'boy', 'brother', 'car', 'cat', 'child', 'cold', 'court', 'deaf', 'dog', 'expensive', 'family', 'fast', 'father', 'friend', 'gift', 'happy', 'hard', 'high', 'horse', 'hospital', 'hot', 'house', 'husband', 'key', 'king', 'laptop', 'library', 'loud', 'man', 'mean', 'money', 'mother', 'new', 'night', 'old', 'paper', 'president', 'quiet', 'school', 'sick', 'sign', 'sister', 'student', 'time', 'tomorrow', 'warm', 'wife', 'window', 'woman', 'year', 'yesterday']

isl_df_train = pd.read_pickle("ISL_dataset_train_FULL.pkl")
isl_df_test = pd.read_pickle("ISL_dataset_test_FULL.pkl")

isl_df_train = isl_df_train[isl_df_train['ISL_GLOSSES'].isin(shared_glosses)]
isl_df_test = isl_df_test[isl_df_test['ISL_GLOSSES'].isin(shared_glosses)]

asl_df = pd.read_pickle("dai_asl.pkl")
asl_df_train, asl_df_test = train_test_split(
    asl_df,
    test_size=0.2,   
    random_state=42,
    shuffle=True,
)

asl_df = asl_df[asl_df['ASL_GLOSSES'].isin(shared_glosses)]
asl_df_train, asl_df_test = train_test_split(
    asl_df,
    test_size=0.2,   
    random_state=42,
    shuffle=True,
    stratify=asl_df['ASL_GLOSSES']
)

gloss_to_idx = {g: i for i, g in enumerate(shared_glosses)}
num_classes = len(shared_glosses)
print(num_classes)
D = 225

d_model = 64
proj_head = nn.Sequential(
    nn.Linear(128, 256),
    nn.ReLU(),
    nn.Linear(256, 128)
).to(device)

isl_proj = nn.Sequential(
    nn.Linear(128, 256),
    nn.ReLU(),
    nn.Linear(256, 128)
).to(device)

asl_proj = nn.Sequential(
    nn.Linear(128, 256),
    nn.ReLU(),
    nn.Linear(256, 128)
).to(device)

# Load your trained transformers
asl_model = TransformerModel(input_dim=D, ntoken = num_classes, d_model = 128, nhead=2, d_hid=256, nlayers= 2, dropout = 0.1, batch_first=False, norm_first=True).to(device)
isl_model = TransformerModel(input_dim=D, ntoken = num_classes, d_model = 128, nhead=2, d_hid=256, nlayers= 2, dropout = 0.1, batch_first=False, norm_first=True).to(device)

asl_model.load_state_dict(torch.load(f'asl_transformer_ASL-HT-10.pt'))
isl_model.load_state_dict(torch.load(f'isl_transformer_ISL-HT-10.pt'))
for param in asl_model.parameters():
    param.requires_grad = True
for param in isl_model.parameters():
    param.requires_grad = True

asl_model.eval()
isl_model.eval()

# Paired dataloader
paired_dataset = PairedDataset(isl_df_train, asl_df_train, shared_glosses, gloss_to_idx)
print('len(paired_dataset)', len(paired_dataset))
paired_loader = DataLoader(paired_dataset, batch_size=512, shuffle=False)
# test_dataset = PairedDataset(isl_df_test, asl_df_test, shared_glosses, gloss_to_idx)
test_dataset = TestDataset(isl_df_test, asl_df_test, shared_glosses, gloss_to_idx)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

# loss_fn = SupervisedInfoNCE(temperature=0.07)
loss_fn = SupConLoss(temperature=0.2)
# all_params = (
#     list(isl_model.parameters()) +
#     list(asl_model.parameters()) +
#     list(isl_proj.parameters()) +
#     list(asl_proj.parameters())
# )

all_params = (
    list(isl_model.parameters()) +
    list(asl_model.parameters()) +
    list(proj_head.parameters()) 
)

optimizer = torch.optim.Adam(all_params, lr=3e-4)
epochs = 500


warmup_epochs = 10
warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs-warmup_epochs, eta_min=1e-6)
scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)
# print(f"warmup total_iters: {warmup.total_iters}")
# print(f"scheduler milestones: {scheduler._milestones}")
experiment_no = 'CL-T-13'
complete_losses = []
test_losses = []
all_inter_cs = []
all_intra_cs = []
all_mean_dist = []
all_ia_top_1, all_ia_top_3 = [], []
all_ai_top_1, all_ai_top_3 = [], []
start = time.time()
for epoch in range(epochs):
    isl_model.train()
    asl_model.train()
    total_loss = 0
    all_isl = []
    all_asl = []
    all_labels = []
    for x_i, x_j, gloss in paired_loader:
        x_i, x_j = x_i.to(device), x_j.to(device)

        z_i, _, _ = isl_model(x_i)   # (B, embed_dim) — ISL embeddings
        z_j, _, _ = asl_model(x_j)   # (B, embed_dim) — ASL embeddings
        z_i = z_i.mean(dim=1)        # (B, embed_dim) — mean pool over sequence
        z_j = z_j.mean(dim=1)        # (B, embed_dim)

        z_i = proj_head(z_i)   # project before loss
        z_j = proj_head(z_j)
        z_i = F.normalize(z_i, p=2, dim=-1)
        z_j = F.normalize(z_j, p=2, dim=-1)

        all_isl.append(z_i)
        all_asl.append(z_j)
        all_labels.append(gloss) 


        features = torch.stack([z_i, z_j], dim=1)
        # shape: (B, 2, D)

        labels = gloss.to(device)
        loss = loss_fn(features, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    all_isl = torch.cat(all_isl, dim=0).cpu().detach().numpy()
    all_asl = torch.cat(all_asl, dim=0).cpu().detach().numpy()
    all_labels = torch.cat(all_labels, dim=0).cpu().detach().numpy() 
    mean_dist, per_class_dists = cross_modal_euclidean(all_isl, all_asl, all_labels)
    print('mean_dist', mean_dist)
    all_mean_dist.append(mean_dist)
    
    features_tr = np.concatenate([all_isl, all_asl], axis=0)
    labels_tr = np.concatenate([all_labels, all_labels], axis=0)
    complete_losses.append(total_loss)
    isl_model.eval()
    asl_model.eval()
    valid_test_glosses = shared_glosses
    
    idx2label = {v: k for k, v in gloss_to_idx.items()}
    results, intra_cs, inter_cs, [IA_top_1, IA_top_3], [AI_top_1, AI_top_3] = retrieve_test(isl_model, asl_model, test_dataset, proj_head, proj_head, epoch, device, idx2label=None)
    all_inter_cs.append(inter_cs)
    all_intra_cs.append(intra_cs)
    
    all_ia_top_1.append(IA_top_1)
    all_ia_top_3.append(IA_top_3)
    all_ai_top_1.append(AI_top_1)
    all_ai_top_3.append(AI_top_3)
    scheduler.step()
    print(f"Epoch {epoch+1} | Loss: {total_loss / len(paired_loader):.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
    if ((epoch+1)%100 == 0) or (epoch == 0): 
        plot_tsne(isl_model, asl_model, test_loader, device, gloss_to_idx, epoch, experiment_no,
                    title=f"t-SNE at Epoch {epoch+1}")



mins = (time.time() - start) / 60
print(experiment_no)

fig2, ax2 = plt.subplots()
# Loss plot
ax2.plot(range(epochs), complete_losses, c='mediumvioletred')
ax2.set_title(f"{experiment_no}, Training Loss")
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Training Loss")
fig2.savefig(f"CL_Graphs_Final/Exp. {experiment_no}, Training Loss.png")

# Clear axes (not figure)
ax2.clear()

# Accuracy plot

ax2.plot(range(epochs), all_ia_top_3, c="#9087d3bb", label = 'I-A Top-3')
ax2.plot(range(epochs), all_ai_top_3, c="#71b7dac4", label = 'A-I Top-3')
ax2.plot(range(epochs), all_ia_top_1, c="#54468eff", label = 'I-A Top-1')
ax2.plot(range(epochs), all_ai_top_1, c="#5cbcdc", label = 'A-I Top-1')
ax2.set_title(f"{experiment_no}, Translation Top-k Accuracies")
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Accuracy")
ax2.legend()
fig2.savefig(f"CL_Graphs_Final/Exp. {experiment_no}, Accuracy.png")

ax2.clear()

# Accuracy plot
ax2.plot(range(epochs), all_inter_cs, c="#c278c4ff", label = 'Inter')
ax2.plot(range(epochs), all_intra_cs, c="#4da16cff", label = 'Intra')
ax2.set_title(f"{experiment_no}, Inter and Intra Cluster Similarities")
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Cosine Similarities")
ax2.legend()
fig2.savefig(f"CL_Graphs_Final/Exp. {experiment_no}, Cluster Sims.png")

ax2.clear()

ax2.plot(range(epochs), all_mean_dist, c="#eb662cff")
ax2.set_title(f"{experiment_no}, Mean distance between ISL and ASL clusters")
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Euclidean Distance")
fig2.savefig(f"CL_Graphs_Final/Exp. {experiment_no}, Mean dists.png")
mins = (time.time() - start) / 60
print(experiment_no)
print(f"Run finished in {mins:.2f} mins")

# rsync -r va452@ogg.cs.bath.ac.uk:/mnt/fast0/va452/ISL_Graphs_Transformer_69 /Users/vrushaliashok/sign_language
# https://par.nsf.gov/servlets/purl/10393836