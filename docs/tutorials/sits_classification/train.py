'''Training script to complete.
'''
import os
import pprint
import argparse
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import yaml

from dataset import PixelSetData, Padding
from models.transformer.transformer import Transformer
from models.classifiers import ShallowClassifier

from myutils.focal_loss import FocalLoss
from myutils.utils import get_flops, get_params


def main(cfg):
    os.makedirs(os.path.dirname(cfg['res_dir']), exist_ok=True)

    with open(os.path.join(cfg['res_dir'], 'train_config.yaml'), 'w') as file:
        yaml.dump(cfg, file)

    dataset = PixelSetData(cfg['data_folder'])
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [1 - cfg['val_split'], cfg['val_split']])
    n_classes = len(np.unique(dataset.labels))
    padding = Padding(pad_value=cfg['pad_value'])

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg['batch_size'],
        shuffle=True,
        num_workers=cfg.get('num_workers', 0),
        collate_fn=padding.pad_collate,
        pin_memory=(cfg.get('device', 'cpu') == 'mps' and torch.mps.is_available()),
    )

    val_data_loader = torch.utils.data.DataLoader(
        val_dataset,
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

    input_shape = (1, cfg['max_len'], cfg['n_channels'], cfg['n_pixels'])
    #flops = get_flops(encoder, input_shape)
    params = get_params(encoder)

    print("Total number of parameters: {:.2f}M".format(params / 10**6))
    #print("Total number of FLOPs: {:.2f}M".format(flops / 10**6))

    classifier = ShallowClassifier(
        d_input=cfg['d_model'],
        d_inner=cfg['classifier']['d_inner'],
        n_classes=n_classes)

    lr = float(cfg['lr'])
    epochs = cfg['epochs']
    optimizer = optim.Adam(list(encoder.parameters()) + list(classifier.parameters()), lr=lr)
    criterion = FocalLoss(cfg['loss']['gamma'])

    if cfg['device'] == 'mps' and torch.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    encoder = encoder.to(device)
    classifier = classifier.to(device)

    metrics = {
        'train': {
            'loss': [],
            'acc': []
        },
        'val': {
            'loss': [],
            'acc': []
        }
    }

    best_val_loss = np.inf

    for epoch in range(epochs):

        encoder.train()
        classifier.train()
        train_loss, train_acc, val_loss, val_acc = 0, 0, 0, 0

        for data, doys, labels in tqdm(train_data_loader, desc=f"Epoch {epoch + 1}"):
            data = data.to(device)
            doys = doys.to(device)
            labels = labels.to(device)

            z, _ = encoder(data, doys)
            logits = classifier(z)
            loss = criterion(logits, labels)
            pred = torch.argmax(logits, dim=1)
            accuracy = (pred == labels).float().mean().item() * 100

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item() / len(train_data_loader)
            train_acc += accuracy / len(train_data_loader)

        print("Train metrics - Loss: {:.4f} - Acc: {:.2f}".format(train_loss, train_acc))
        metrics['train']['loss'].append(train_loss)
        metrics['train']['acc'].append(train_acc)

        encoder.eval()
        classifier.eval()
        for data, doys, labels in tqdm(val_data_loader, desc=f"Validation"):
            data = data.to(device)
            doys = doys.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                z, _ = encoder(data, doys)
                logits = classifier(z)
            loss = criterion(logits, labels)
            pred = torch.argmax(logits, dim=1)
            accuracy = (pred == labels).float().mean().item() * 100

            val_loss += loss.item() / len(val_data_loader)
            val_acc += accuracy / len(val_data_loader)

        print("Val metrics - Loss: {:.4f} - Acc: {:.2f}".format(val_loss, val_acc))
        metrics['val']['loss'].append(val_loss)
        metrics['val']['acc'].append(val_acc)

        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            torch.save({'epoch': epoch, 
                        'encoder': encoder.state_dict(),
                        'classifier': classifier.state_dict(),
                        'optimizer': optimizer.state_dict()},
                        os.path.join(cfg['res_dir'], 'best_model.pth.tar'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_file', type=str,
                        help='Path to the training config.')
   
    parser = parser.parse_args()

    with open(parser.cfg_file, 'r') as file:
        cfg = yaml.safe_load(file)

    pprint.pprint(cfg)
    main(cfg)

    