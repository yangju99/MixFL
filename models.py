import torch
from torch import nn
from dgl.nn.pytorch import GATv2Conv
import pdb 
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        """
        alpha: scalar (e.g., 5.0 for class 1 emphasis) or Tensor([alpha_0, alpha_1])
        gamma: focusing parameter
        reduction: 'mean', 'sum', or 'none'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)  

        if isinstance(self.alpha, torch.Tensor):
            at = self.alpha.to(logits.device)[targets]
        else:
            at = self.alpha

        focal_loss = at * ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss



class CommentModel(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(CommentModel, self).__init__()
        self.embedder = nn.Linear(in_dim, out_dim) 
    def forward(self, paras: torch.tensor): 
        """
        Input:
            paras: mu with length of event_num
        """
        return self.embedder(paras)


class CodeModel(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(CodeModel, self).__init__()
        self.embedder = nn.Linear(in_dim, out_dim) 
    def forward(self, paras: torch.tensor): 
        """
        Input:
            paras: mu with length of event_num
        """
        return self.embedder(paras)


class MultiSourceEncoder(nn.Module):

    def __init__(
        self,
        device,
        code_out_dim=512,
        comment_out_dim=512,
        fuse_dim=512,
        debug: bool = False,
        **kwargs
    ):
        super().__init__()
        self.device = device
        self.debug = debug

        # kwargs['code_dim'] / ['comment_dim']: {model_name: input_dim}
        code_dim = kwargs['code_dim']
        comment_dim = kwargs['comment_dim']

        self.model_names = list(code_dim.keys())

        # per-model encoders
        self.code_models    = nn.ModuleDict({m: CodeModel(code_dim[m],      code_out_dim)     for m in self.model_names})
        self.comment_models = nn.ModuleDict({m: CommentModel(comment_dim[m], comment_out_dim) for m in self.model_names})

        # fuse (per model): [code_out + comment_out] → fuse_dim → GLU (halve)
        if fuse_dim % 2 != 0:
            fuse_dim += 1  
        self.fusers   = nn.ModuleDict({m: nn.Linear(code_out_dim + comment_out_dim, fuse_dim) for m in self.model_names})
        self.activate = nn.GLU()

        # per-model feature dim after GLU
        self.per_model_feat_dim = fuse_dim // 2

        # --- Attention pooling over models ---
        # f_m ∈ R^{N_total × F} → score_m ∈ R^{N_total × 1}, softmax across M
        self.attn_scorer = nn.Linear(self.per_model_feat_dim, 1, bias=True)

        self.feat_out_dim = self.per_model_feat_dim

    def forward(self, per_model_graphs: dict):
        """
        per_model_graphs: { model_name: batched_dgl_graph }
        """
        feats = []
        total_nodes = None

        # 1) per-model feature 추출
        for m in self.model_names:
            g = per_model_graphs[m]
            code = self.code_models[m](g.ndata["code_vector"])        # [N_total, code_out_dim]
            cmt  = self.comment_models[m](g.ndata["comment_vector"])  # [N_total, comment_out_dim]
            fused = self.fusers[m](torch.cat([code, cmt], dim=-1))    # [N_total, fuse_dim]
            fused = self.activate(fused)                              # [N_total, F]
            feats.append(fused)

        # 2) Attention Pooling over model dimension
        # feats: list of M tensors [N_total, F] → stack: [M, N_total, F]
        feats_stacked = torch.stack(feats, dim=0)

        #  [M, N_total, 1]
        scores = self.attn_scorer(feats_stacked)  #
        # softmax over model axis (dim=0): [M, N_total, 1]
        attn = torch.softmax(scores, dim=0)

        # 가중합: (attn * feats_stacked) sum over M → [N_total, F]
        feat = (attn * feats_stacked).sum(dim=0)

        return feat  # [N_total, F]  (self.feat_out_dim = F)


class FullyConnected(nn.Module):
    def __init__(self, in_dim, out_dim, linear_sizes):
        super(FullyConnected, self).__init__()
        layers = []
        for i, hidden in enumerate(linear_sizes):
            input_size = in_dim if i == 0 else linear_sizes[i-1]
            layers += [nn.Linear(input_size, hidden), nn.ReLU()]
        layers += [nn.Linear(linear_sizes[-1], out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.net(x)


import numpy as np
class MainModel(nn.Module):
    def __init__(self, device, debug=False, **kwargs):
        super(MainModel, self).__init__()

        self.device = device

        self.encoder = MultiSourceEncoder(device, debug=debug, **kwargs)

        self.classifier = FullyConnected(self.encoder.feat_out_dim, 2, kwargs['classification_hiddens']).to(device)

        self.criterion = FocalLoss(gamma=2.0) 
        self.get_prob = nn.Softmax(dim=-1)

    def forward(self, graph, fault_indexs, node_counts):
        """
        Pointwise loss version.
        """
        embeddings = self.encoder(graph)  

        logits = self.classifier(embeddings)

        y_true = torch.zeros(logits.size(0), dtype=torch.long, device=self.device)

        offset = 0
        for faults, count in zip(fault_indexs, node_counts):
            for idx in faults:
                y_true[offset + idx] = 1
            offset += count

        loss = self.criterion(logits, y_true)

        node_probs = self.get_prob(logits.detach()).cpu().numpy()

        y_pred = self.inference(node_probs, node_counts)

        return {
            'loss': loss,
            'y_pred': y_pred
        }


    def inference(self, node_probs, node_counts):
        """
        node_probs: [total_nodes, 2]
        node_counts: list of node counts per graph in batch
        Returns:
            list of ranked indices per graph
        """
        fault_probs = node_probs[:, 1]  
        results = []
        start = 0
        for count in node_counts:
            sub_probs = fault_probs[start:start+count]
            ranked = sub_probs.argsort()[::-1].tolist()
            results.append(ranked)  
            start += count
        return results


