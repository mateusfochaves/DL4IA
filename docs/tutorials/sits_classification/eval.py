'''Test script.
'''

import torch
import torch.nn.functional as F
import numpy as np
import json
import matplotlib.pyplot as plt
from dataset import PixelSetData, Padding # pad_collate
from models.transformer.transformer import Transformer
from models.classifiers import ShallowClassifier
import torch.optim as optim
import pdb
import os
from tqdm import tqdm
import yaml
import sys
import pickle as pkl
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from myutils.utils import get_flops, get_params
from fvcore.nn import FlopCountAnalysis

if len(sys.argv) < 2:
    print("Usage: python eval.py <results_folder>")
    sys.exit(1)
results_folder = sys.argv[1]

print("RESULTS FOLDER =", results_folder)
print("EXISTS?", os.path.exists(results_folder))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

with open(os.path.join(results_folder, 'train_config.yaml'), 'r') as file:
    cfg = yaml.safe_load(file)

for k, v in cfg.items():
    print(k + ': ' + str(v))

dataset = PixelSetData(cfg['data_folder'], set='test')

n_classes = len(np.unique(dataset.labels))

padding = Padding(pad_value=cfg['pad_value'])

data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=cfg['batch_size'],
    shuffle=False,
    num_workers=cfg.get('num_workers', 0),
    collate_fn=padding.pad_collate,
    pin_memory=(cfg.get('device', 'cpu') == 'mps' and torch.mps.is_available()),
)

encoder = Transformer(
    n_channels=cfg['n_channels'],
    n_pixels=cfg['n_pixels'],
    d_model=cfg['d_model'],
    d_inner=cfg['d_inner'],
    n_head=cfg['n_head'],
    d_k=cfg['d_k'],
    d_v=cfg['d_v'],
    dropout=cfg['dropout'],
    pad_value=cfg['pad_value'],
    scale_emb_or_prj=cfg['scale_emb_or_prj'],
    n_position=cfg['pos_embedding']['n_position'],
    T=cfg['pos_embedding']['T'],
    return_attns=cfg['return_attns'],
    learnable_query=cfg['learnable_query'],
    spectral_indices_embedding=cfg['spectral_indices'],
    channels=cfg['channels'],
    compute_values=cfg['compute_values']
)

classifier = ShallowClassifier(
    d_input=cfg['d_model'],
    d_inner=cfg['classifier']['d_inner'],
    n_classes=n_classes)


if cfg['device'] == 'mps' and torch.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

encoder = encoder.to(device)
classifier = classifier.to(device)
checkpoint = torch.load(os.path.join(results_folder, 'best_model.pth.tar'), map_location=device)
encoder.load_state_dict(checkpoint['encoder'])
classifier.load_state_dict(checkpoint['classifier'])
del checkpoint

encoder.eval()
classifier.eval()

n_params_encoder = count_parameters(encoder)
n_params_classifier = count_parameters(classifier)
n_params_total = n_params_encoder + n_params_classifier

print("=== PARAMETERS ===")
print("Encoder params:", n_params_encoder)
print("Classifier params:", n_params_classifier)
print("Total params:", n_params_total)

def compute_flops(encoder, classifier, data_loader, device):
    encoder.eval()
    classifier.eval()

    data, doys, labels = next(iter(data_loader))
    data = data.to(device)
    doys = doys.to(device)

    class FullModel(torch.nn.Module):
        def __init__(self, encoder, classifier):
            super().__init__()
            self.encoder = encoder
            self.classifier = classifier

        def forward(self, data, doys):
            z, _ = self.encoder(data, doys)
            logits = self.classifier(z)
            return logits

    full_model = FullModel(encoder, classifier).to(device)
    full_model.eval()

    flops = FlopCountAnalysis(full_model, (data, doys))
    return flops.total()

y_pred = []
y_true = []
for data, doys, labels in tqdm(data_loader, desc=f"Test"):
    data = data.to(device)
    doys = doys.to(device)
    labels = labels.to(device)
    batch_size = data.shape[0]

    with torch.no_grad():
        z, _ = encoder(data, doys)
        logits = classifier(z)
    loss = F.cross_entropy(logits, labels)
    pred = torch.argmax(logits, dim=-1)
    accuracy = (pred == labels).float().mean()

    y_pred.extend(pred.tolist())
    y_true.extend(labels.tolist())


f1_score_ = f1_score(y_true, y_pred, average=None)
f1_score_ = [round(float(x), 2) for x in f1_score_]

metrics = {
    "f1_score": f1_score_,
    "avg_f1_score": f1_score(y_true, y_pred, average="macro"),
    "accuracy": accuracy_score(y_true, y_pred)
}

cm = confusion_matrix(y_true, y_pred)

with open(os.path.join(results_folder, 'test_metrics.yaml'), 'w') as file:
    yaml.dump(metrics, file)

pkl.dump(cm, open(os.path.join(results_folder, 'conf_mat.pkl'), 'wb'))

print("=== TEST METRICS ===")
print("Accuracy:", metrics["accuracy"])
print("Macro F1:", metrics["avg_f1_score"])
print("F1 per class:", metrics["f1_score"])
try:
    total_flops = compute_flops(encoder, classifier, data_loader, device)
    print("=== FLOPs ===")
    print("Total FLOPs:", total_flops)
except Exception as e:
    print("Could not compute FLOPs:", e)
    total_flops = None