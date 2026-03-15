import torch
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode

import numpy as np

from typing import Union, Tuple, Optional, List


def dates2doys(dates: list[str]):
    ''' convert a list of dates (list of str) 
    to a list of days of year. Referecing them from 0 to 365 (bisextile)
    '''
    doys = []
    for date in dates: 
        month = date[5:7]
        day = date[8:]
        doy = int(month) * 30 + int(day)
        doys.append(doy)

    return torch.tensor(doys).long()



def pad_tensor(x: torch.Tensor, l: int, pad_value=0.):
    ''' Adds padding to a tensor.
    '''
    padlen = l - x.shape[0]
    pad = [0 for _ in range(2 * len(x.shape[1:]))] + [0, padlen]
    return F.pad(x, pad=pad, value=pad_value)


def fill_ts(ts: torch.Tensor, doys: torch.Tensor, full_doys: torch.Tensor):
    ''' Fill the gaps in a time series with NaN values.
    Args:
        ts: time series with missing data
        doys: days of year of the time series
        full_doys: complete list of days of year (including missing dates)
    '''
    full_length = len(full_doys)
    ts = pad_tensor(ts, full_length, pad_value=torch.nan)
    missing_doys = torch.tensor(list(
        set(full_doys.tolist()) - set(doys.tolist())
    ))
    missing_doys, _ = missing_doys.sort()
    doys = torch.cat((doys, missing_doys))
    doys, indices = doys.sort()
    indices = indices.view(-1, 1, 1).repeat(1, ts.shape[1], ts.shape[2])
    ts = torch.gather(ts, index=indices, dim=0)
    return ts


def get_params(model: torch.nn.Module):
    '''TODO: compute the number of trainable parameters of a model.
    '''
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_flops(model, inp: Union[torch.Tensor, Tuple], with_backward=False):
    '''Credit: https://alessiodevoto.github.io/Compute-Flops-with-Pytorch-built-in-flops-counter/
    '''
    istrain = model.training
    model.eval()
    
    inp = inp if isinstance(inp, torch.Tensor) else torch.randn(inp)

    flop_counter = FlopCounterMode(mods=model, display=False, depth=None)
    with flop_counter:
        if with_backward:
            model(inp).sum().backward()
        else:
            model(inp)
    total_flops =  flop_counter.get_total_flops()
    if istrain:
        model.train()
    return total_flops

def rgb_render(
    data: np.ndarray,
    clip: int = 2,
    bands: Optional[List[int]] = None,
    norm: bool = True,
    dmin: Optional[np.ndarray] = None,
    dmax: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Prepare data for visualization with matplot lib, taken (with minor modifications) from the sensorsio repo by Julien Michel
    Source: https://framagit.org/jmichel-otb/sensorsio/-/blob/master/src/sensorsio/utils.py?ref_type=heads
    License: Apache License, Version 2.0

    :param data: nd_array of shape [bands, w, h]
    :param clip: clip percentile (between 0 and 100). Ignored if norm is False
    :bands: List of bands to extract (len is 1 or 3 for RGB)
    :norm: If true, clip a percentile at each end

    :returns: a tuple of data ready for matplotlib, dmin, dmax
    """
    if bands is None:
        bands = [2, 1, 0]
    assert len(bands) == 1 or len(bands) == 3
    assert 0 <= clip <= 100

    # Extract bands from data
    data_ready = np.take(data, bands, axis=0)
    out_dmin = None
    out_dmax = None
    # If normalization is on
    if norm:
        # Rescale and clip data according to percentile
        if dmin is None:
            out_dmin = np.percentile(data_ready, clip, axis=(1, 2))
        else:
            out_dmin = dmin
        if dmax is None:
            out_dmax = np.percentile(data_ready, 100 - clip, axis=(1, 2))
        else:
            out_dmax = dmax
        data_ready = np.clip((data_ready.transpose(1, 2, 0) - out_dmin) / (out_dmax - out_dmin), 0, 1)

    else:
        data_ready.transpose(1, 2, 0)

    # Strip of one dimension if number of bands is 1
    if data_ready.shape[-1] == 1:
        data_ready = data_ready[:, :, 0]

    return data_ready, out_dmin, out_dmax

import torch
from tqdm import tqdm

def mean_attention(model, dataset, select_class=None, batch_size=64, pad_value=0, max_len=24, device=None):
    """
    Retorna Tensor [H, T_out] (1D por head), compatível com o plot do TP.

    Funciona tanto para attn quadrada [T,T] quanto para attn com learnable query:
    [Q,K] (ex.: Q=1, K=T).
    """
    from dataset import Padding

    model.eval()
    if device is None:
        device = next(model.parameters()).device

    padding = Padding(pad_value=pad_value)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=padding.pad_collate,
    )

    sum_mask = None      # [H, T_out]
    count_mask = None    # [H, T_out]
    T_out = None

    with torch.no_grad():
        for data, doys, labels in tqdm(loader, desc="Test"):
            # filtra por classe
            if select_class is not None:
                m = (labels == select_class)
                if m.sum().item() == 0:
                    continue
                data = data[m]
                doys = doys[m]

            data = data.to(device)
            doys = doys.to(device).long()  # [B, T_doys]

            _, attns = model(data, doys)

            # normaliza attns -> tensor [L,B,H,*,*] ou [B,H,*,*]
            if isinstance(attns, (list, tuple)):
                # lista por layer: cada item pode ser [B,H,Q,K] ou [B,Q,K] se H=1
                a0 = attns[0]
                if a0.dim() == 3:  # [B,Q,K] -> add H=1
                    attns = [a.unsqueeze(1) for a in attns]  # [B,1,Q,K]
                attns = torch.stack(attns, dim=0)           # [L,B,H,Q,K]

            if attns.dim() == 5:        # [L,B,H,Q,K]
                A = attns.mean(dim=0)   # [B,H,Q,K]
            elif attns.dim() == 4:      # [B,H,Q,K]
                A = attns
            else:
                raise ValueError(f"Formato inesperado de attns: {attns.shape}")

            B, H, Q, K = A.shape

            # define T_out pelo eixo de KEYS (é isso que vira eixo-x no plot)
            if T_out is None:
                T_out = min(K, max_len)  # max_len só como limite de segurança
                sum_mask = torch.zeros((H, T_out), device=device, dtype=A.dtype)
                count_mask = torch.zeros((H, T_out), device=device, dtype=torch.float32)

            # timesteps válidos (no TP normalmente padding é 0)
            valid = (doys != 0)  # [B, T_doys]

            for b in range(B):
                idx = torch.where(valid[b])[0]          # índices válidos no doys
                if idx.numel() == 0:
                    continue

                # idx tem que existir no eixo das KEYS (K)
                idx = idx[idx < K]
                if idx.numel() == 0:
                    continue

                tb = int(min(idx.numel(), T_out))
                idx = idx[:tb]

                Ab = A[b]  # [H,Q,K]

                if Q == K:
                    # atenção quadrada: dá pra cortar queries e keys
                    Ab2 = Ab[:, idx][:, :, idx]         # [H,tb,tb]
                    mask_1d = Ab2.mean(dim=1)           # média sobre queries -> [H,tb]
                else:
                    # learnable query / CLS: só corta nas KEYS
                    Ab2 = Ab[:, :, idx]                 # [H,Q,tb]
                    mask_1d = Ab2.mean(dim=1)           # média sobre Q -> [H,tb]

                sum_mask[:, :tb] += mask_1d
                count_mask[:, :tb] += 1.0

    if sum_mask is None or count_mask.sum().item() == 0:
        raise ValueError(f"Nenhuma amostra encontrada para select_class={select_class}")

    mean_mask = sum_mask / torch.clamp(count_mask, min=1.0)  # [H,T_out]

    # opcional: normaliza por head (deixa comparável e limita valores)
    mean_mask = mean_mask / (mean_mask.sum(dim=1, keepdim=True) + 1e-12)

    return mean_mask.detach().cpu()