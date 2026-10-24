import os

import torch
import torch.nn as nn

from typing import Dict
import numpy as np
from copy import deepcopy

INDEX_POSITIONS_TEXT = {
    'top1': [11],
    'top2': [10, 11],
    'top3': [9, 10, 11],
    'bottom': [0, 1, 2, 3],
    'mid': [4, 5, 6, 7],
    'up': [8, 9, 10, 11],
    'half-up': [6, 7, 8, 9, 10, 11],
    'half-bottom': [0, 1, 2, 3, 4, 5],
    'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]}


INDEX_POSITIONS_VISION = {
    'ViT-B/16': {
        'top': [11],
        'top3': [9, 10, 11],
        'bottom': [0, 1, 2, 3],
        'mid': [4, 5, 6, 7],
        'up': [8, 9, 10, 11],
        'half-up': [6, 7, 8, 9, 10, 11],
        'half-bottom': [0, 1, 2, 3, 4, 5],
        'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]},
    'ViT-B/32': {
        'bottom': [0, 1, 2, 3],
        'mid': [4, 5, 6, 7],
        'up': [8, 9, 10, 11],
        'half-up': [6, 7, 8, 9, 10, 11],
        'half-bottom': [0, 1, 2, 3, 4, 5],
        'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]},

    'ViT-L/14': {
        'bottom': [0, 1, 2, 3],
        'mid': [4, 5, 6, 7],
        'up': [8, 9, 10, 11],
        'half-up': [6, 7, 8, 9, 10, 11],
        'half-bottom': [0, 1, 2, 3, 4, 5],
        'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]}
}


def mark_only_lora_as_trainable(model: nn.Module, bias: str = 'none') -> None:
    for n, p in model.named_parameters():
        if 'lora_' not in n:
            p.requires_grad = False
    if bias == 'none':
        return
    elif bias == 'all':
        for n, p in model.named_parameters():
            if 'bias' in n:
                p.requires_grad = True
    elif bias == 'lora_only':
        for m in model.modules():
            if getattr(m, 'is_lora_layer', False) and \
                    hasattr(m, 'bias') and \
                    m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


def lora_state_dict(model: nn.Module, bias: str = 'none') -> Dict[str, torch.Tensor]:
    my_state_dict = model.state_dict()
    if bias == 'none':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k}
    elif bias == 'all':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k or 'bias' in k}
    elif bias == 'lora_only':
        to_return = {}
        for k in my_state_dict:
            if 'lora_' in k:
                to_return[k] = my_state_dict[k]
                bias_name = k.split('lora_')[0]+'bias'
                if bias_name in my_state_dict:
                    to_return[bias_name] = my_state_dict[bias_name]
        return to_return
    else:
        raise NotImplementedError


def get_lora_parameters(model, bias='none'):
    params = []
    for name, param in model.named_parameters():
        if bias == 'none':
            if 'lora_' in name:
                params.append(param)
        elif bias == 'all':
            if 'lora_' in name or 'bias' in name:
                params.append(param)
        elif bias == 'lora_only':
            if 'lora_' in name:
                params.append(param)
                bias_name = name.split('lora_')[0] + 'bias'
                if bias_name in model.state_dict():
                    bias_param = dict(model.named_parameters())[bias_name]
                    params.append(bias_param)
        else:
            raise NotImplementedError
    return params


import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import math




class LoRALayer(torch.nn.Module):
    """
    MOT cap LoRA (A, B) DUNG CHUNG cho tat ca cac task.

    Vong doi cua cap A/B trong continual learning:
      - Task t:  A da duoc chon (ngau nhien o task 0, tu SVD basis o task > 0)
                 va DONG BANG; chi B duoc huan luyen.
      - Khi chuyen sang task t+1:  gop B_t @ A_t vao `old_weight`,
                 chon A moi tu SVD basis, reset B ve 0.

    Nho vay so tham so LoRA khong tang theo so task, con `old_weight` giu lai
    toan bo bien doi da hoc (tuong duong W_t = W_0 + sum_t B_t A_t).
    """

    # Marker de cac ham generic nhan biet day la lop LoRA cua repo nay
    # (lop LoRALayer trong layers.py chi la class tron, khong phai nn.Module).
    is_lora_layer = True

    def __init__(self, in_dim, out_dim, rank, alpha, pos='', layer=0):
        super().__init__()
        self.in_dim, self.out_dim, self.rank = in_dim, out_dim, rank
        self.alpha = alpha
        self.pos = pos
        self.layer = layer

        # ---- Cap A/B dung chung (khong phai ModuleList theo task) ----
        # Hinh dang A: (rank, in_dim)  ->  B @ A: (out_dim, in_dim)
        self.lora_A = nn.Parameter(torch.empty(self.rank, self.in_dim))
        self.lora_B = nn.Parameter(torch.zeros(self.out_dim, self.rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B khoi tao = 0 => san pham B@A = 0, mo hinh ban dau khong doi.

        # Moc chuan cua A (norm cua A ngau nhien ban dau), dung lam thang chuan
        # cho moi A sinh ra o task sau -> tranh suy giam theo cap so nhan.
        self.register_buffer(
            "initial_A_norm", self.lora_A.detach().norm().clone()
        )

        # Tong cac B_t A_t da hoc -> phan bien doi "dong bang" cua backbone.
        self.register_buffer("old_weight", torch.zeros(self.out_dim, self.in_dim))

        # Thong ke activation de chon SVD basis.
        self.cur_matrix = torch.zeros(self.in_dim, self.in_dim)
        self.n_cur_matrix = 0

        self.task_id = 0
        
    def update(self, task_id):
        """
        Dong bo task id cua layer.
        """
        self.task_id = task_id
    def update_grad_before_train(self, task_id):
        """
        Chu cho task `task_id` (> 0):
          1. gop B@A cua task truoc vao `old_weight` (giu nguyen hanh vi dau ra);
          2. dung A moi tu SVD basis cua task truoc;
          3. reset B ve 0 de task moi bat dau tu ham dang nhat.
        """
        if task_id == 0:
            return

        with torch.no_grad():
            a_scale = float(os.environ.get("LORA_A_SCALE", "1.0"))
            a_pow = float(os.environ.get("LORA_A_POW", "1.0"))

            A = self.lora_A
            B = self.lora_B

            # (1) Gop ph bien doi da hoc -> giu nguyen output tren task cu.
            #     Day la DIEM DUY NHAT duoc gop B@A cho lop nay.
            self.old_weight.add_(B @ A)

            # (2) Chon A moi trong khong gian SVD basis.
            basis = getattr(self, 'feature_list', None)
            if basis is None or basis.numel() == 0:
                # Chua co basis -> giu A hien tai, chi reset B.
                B.zero_()
                return

            basis = basis.to(device=A.device, dtype=A.dtype)
            coeffs = torch.randn(basis.shape[1], self.rank,
                                 device=A.device, dtype=A.dtype)
            new_A = (basis @ coeffs).T                      # (rank, in_dim)

            # Chuan hoa ve norm cua A ban dau, roi co theo (t+1)^p de kiem soat
            # muc thich ngi cua task moi.
            target_norm = (self.initial_A_norm.to(device=A.device, dtype=A.dtype)
                           / ((task_id + 1) ** a_pow) * a_scale)
            new_A = new_A / new_A.norm().clamp_min(1e-12) * target_norm

            A.copy_(new_A)

            # (3) Reset B -> task moi khoi dau tu ham dang nhat.
            B.zero_()
    def update_grad_after_train(self, task_id):
        """
        Cuoi task: SVD ma tran Gram cua activation de chon khong gian (basis)
        ma A cua task ke tiep se nam trong do.

        Nguyen tac: bo qua r huong troi (da bi `old_weight` chiem), giu lai
        cac huong con lai de task moi co cho de hoc thu moi.
        """
        with torch.no_grad():
            activation = self.cur_matrix
            U, S, Vh = torch.linalg.svd(activation, full_matrices=False)

            sval_total = S.sum().clamp_min(1e-12)
            sval_ratio = S / sval_total
            k = torch.arange(1, self.in_dim + 1, device=S.device) - 1
            k = (self.in_dim - k.float()) / self.in_dim
            result = ((task_id + 1) * (1 - torch.cumsum(sval_ratio, dim=-1))
                      - self.alpha * k)
            r = int(torch.argmin(result))

            # Giu nguyen semantics ban goc, nhung chan trong hop basis rong.
            start = max(r, 1) + 1
            if start >= U.shape[1]:
                start = max(U.shape[1] - 1, 1)
            self.feature_list = U[:, start:].contiguous()
            self.feature_mat = self.feature_list @ self.feature_list.T

            wandb.log({f'feature_matrix/{self.layer}_{self.pos}':
                       self.feature_list.shape[1]})


    def forward(self, x, get_cur_feat=False):

        if get_cur_feat:
            # Ma tran Gram trung bin cua activation dau vao cua lop LoRA.
            self.cur_matrix = self.cur_matrix.to(x.device)
            self.cur_matrix = (self.cur_matrix * self.n_cur_matrix
                               + torch.bmm(x.detach().permute(0, 2, 1),
                                           x.detach()).sum(dim=0)
                               ) / (self.n_cur_matrix
                                    + x.shape[0] * x.shape[1])
            self.n_cur_matrix += x.shape[0] * x.shape[1]

        # Cap A/B dung chung: old_weight chua tong B_t A_t da hoc,
        # con lora_B @ lora_A la ph bien doi cua task dang train.
        result = F.linear(x, self.old_weight + self.lora_B @ self.lora_A)

        return result



class QkvWithLoRA(torch.nn.Module):
    def __init__(self, qkv, rank, alpha, layer):
        super().__init__()
        self.qkv = qkv
        self.dim = qkv.in_features
        # self.lora_q = LoRALayer(self.dim, self.dim, rank, alpha,rate,args)
        self.lora_k = LoRALayer(self.dim, self.dim, rank, alpha, pos='k',layer=layer)
        self.lora_v = LoRALayer(self.dim, self.dim, rank, alpha, pos='v',layer=layer)
        self.layer = layer

    def update(self, task_id):
        # self.lora_q.update(task_id)
        self.lora_k.update(task_id)
        self.lora_v.update(task_id)
    def update_grad(self, task_id, many):
        if many == 'before':
            self.lora_k.update_grad_before_train(task_id)
            self.lora_v.update_grad_before_train(task_id)
        elif many == 'after':
            self.lora_k.update_grad_after_train(task_id)
            self.lora_v.update_grad_after_train(task_id)
    def forward(self, x, get_cur_feat=False):

        qkv = self.qkv(x)
        qkv[:, :, self.dim:2*self.dim] += self.lora_k(x, get_cur_feat=get_cur_feat)
        qkv[:, :, -self.dim:] += self.lora_v(x, get_cur_feat=get_cur_feat)

        return qkv

 
class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)

    def update(self, task_id):
        self.lora.update(task_id)

def apply_lora(args, model):
    """
    Gan LoRA vao cac block cua ViT.

    Vi cap A/B duoc DUNG CHUNG cho moi task, chi can gan MOT LAN (task 0).
    Cac task sau khong tao lai module, chi dong bo task id cho cac lop da gan.
    """
    rank = args.r
    alpha = args.alpha
    list_lora_layers = []

    if args.task_id != 0:
        # Da gan o task 0 roi -> chi dong bo task id.
        for block in model.blocks:
            if hasattr(block.attn, 'qkv') and hasattr(block.attn.qkv, 'lora_k'):
                block.attn.qkv.lora_k.task_id = args.task_id
                block.attn.qkv.lora_v.task_id = args.task_id
            if hasattr(block.attn, 'proj') and hasattr(block.attn.proj, 'lora'):
                block.attn.proj.lora.task_id = args.task_id
            if hasattr(block.mlp, 'fc1') and hasattr(block.mlp.fc1, 'lora'):
                block.mlp.fc1.lora.task_id = args.task_id
                block.mlp.fc2.lora.task_id = args.task_id
        return list_lora_layers

    for i, block in enumerate(model.blocks):
        block.attn.qkv = QkvWithLoRA(block.attn.qkv, rank=rank, alpha=alpha,
                                     layer=i)
        if 'o' in args.params:
            block.attn.proj = LinearWithLoRA(block.attn.proj, rank, alpha)
        if 'mlp' in args.params:
            block.mlp.fc1 = LinearWithLoRA(block.mlp.fc1, rank, alpha)
            block.mlp.fc2 = LinearWithLoRA(block.mlp.fc2, rank, alpha)

    return list_lora_layers

def update_grad(taskid, model, many):
    """
    Cap nhat trang thai LoRA cho cac task.

    many == 'before': chuyen sang task moi -> gop B@A vao old_weight,
                      chon A moi tu SVD basis, reset B ve 0.
    many == 'after' : cuoi task -> tinh SVD basis cho task ke tiep.
    """
    for block in model.blocks:
        qkv = getattr(block.attn, 'qkv', None)
        if qkv is not None and hasattr(qkv, 'lora_k'):
            qkv.update_grad(task_id=taskid, many=many)

        # Dong bo task id + gop cho cac lop LoRA khac (proj / fc1 / fc2)
        # neu nguoi dung bat 'o' hoac 'mlp' trong --params.
        for mod in (getattr(block.attn, 'proj', None),
                    getattr(block.mlp, 'fc1', None),
                    getattr(block.mlp, 'fc2', None)):
            if mod is not None and hasattr(mod, 'lora'):
                mod.lora.task_id = taskid
                if many == 'before':
                    mod.lora.update_grad_before_train(taskid)
                elif many == 'after':
                    mod.lora.update_grad_after_train(taskid)

    return


def _collect_lora_layers(model):
    """Thu thap (ten, LoRALayer) cua moi lop LoRA trong model."""
    out = []
    for i, block in enumerate(model.blocks):
        qkv = getattr(block.attn, 'qkv', None)
        if qkv is not None and hasattr(qkv, 'lora_k'):
            out.append((f'layer_{i}.k', qkv.lora_k))
            out.append((f'layer_{i}.v', qkv.lora_v))
        proj = getattr(block.attn, 'proj', None)
        if proj is not None and hasattr(proj, 'lora'):
            out.append((f'layer_{i}.proj', proj.lora))
        fc1 = getattr(block.mlp, 'fc1', None)
        if fc1 is not None and hasattr(fc1, 'lora'):
            out.append((f'layer_{i}.fc1', fc1.lora))
        fc2 = getattr(block.mlp, 'fc2', None)
        if fc2 is not None and hasattr(fc2, 'lora'):
            out.append((f'layer_{i}.fc2', fc2.lora))
    return out


def save_lora(args, model, filename=None):
    """
    Luu trang thai LoRA (A, B, old_weight) cua model.

    Vi dung chung MOT cap A/B cho moi task, khong can luu theo tung task:
    chi can A hien tai, B hien tai va `old_weight` (tong B_t A_t da hoc).
    """
    layers = _collect_lora_layers(model)
    weights = {
        name: {
            'lora_A': layer.lora_A.detach().cpu(),
            'lora_B': layer.lora_B.detach().cpu(),
            'old_weight': layer.old_weight.detach().cpu(),
        }
        for name, layer in layers
    }

    metadata = {
        'r': args.r,
        'alpha': args.alpha,
        'encoder': getattr(args, 'encoder', 'both'),
        'params': list(args.params),
        'position': getattr(args, 'position', ''),
    }

    save_data = {'weights': weights, 'metadata': metadata}

    backbone = args.backbone.replace('/', '').replace('-', '').lower()
    save_dir = f'{args.save_path}/{backbone}/seed{args.seed}'
    os.makedirs(save_dir, exist_ok=True)

    fname = filename or getattr(args, 'filename', 'lora')
    save_path = f'{save_dir}/{fname}.pt'
    torch.save(save_data, save_path)
    print(f'LoRA weights saved to {save_path}')


def load_lora(args, model):
    """Nap lai trang thai LoRA da luu (xem `save_lora`)."""
    backbone = args.backbone.replace('/', '').replace('-', '').lower()
    fname = getattr(args, 'filename', 'lora')
    load_path = f'{args.save_path}/{backbone}/seed{args.seed}/{fname}.pt'

    if not os.path.exists(load_path):
        raise FileNotFoundError(f'File {load_path} does not exist.')

    loaded_data = torch.load(load_path, map_location='cpu')
    metadata = loaded_data['metadata']
    if metadata['r'] != args.r:
        raise ValueError(f"r mismatch: expected {args.r}, found {metadata['r']}")
    if metadata['alpha'] != args.alpha:
        raise ValueError(f"alpha mismatch: expected {args.alpha}, found {metadata['alpha']}")

    weights = loaded_data['weights']
    for name, layer in _collect_lora_layers(model):
        if name not in weights:
            continue
        lw = weights[name]
        layer.lora_A.data.copy_(lw['lora_A'].to(layer.lora_A.device))
        layer.lora_B.data.copy_(lw['lora_B'].to(layer.lora_B.device))
        layer.old_weight.data.copy_(lw['old_weight'].to(layer.old_weight.device))

    print(f'LoRA weights loaded from {load_path}')
