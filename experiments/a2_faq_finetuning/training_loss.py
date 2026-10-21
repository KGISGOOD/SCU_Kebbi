"""MNRL-compatible cosine InfoNCE with explicit per-query masking.

Reference objective: SentenceTransformers v2.2.2 MultipleNegativesRankingLoss.
Unlike stock MNRL, hard documents are exposed only to their owner query.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F


class MaskedMultipleNegativesRankingLoss(nn.Module):
    def __init__(self, scale=20.0):
        super().__init__()
        self.scale = float(scale)
        if not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("scale must be finite and positive (temperature = 1 / scale)")

    def forward(self, query_embeddings, document_embeddings, allowed_mask, targets):
        if (query_embeddings.ndim != 2 or document_embeddings.ndim != 2 or
                query_embeddings.shape[0] == 0 or document_embeddings.shape[0] == 0 or
                query_embeddings.shape[1] == 0 or
                query_embeddings.shape[1] != document_embeddings.shape[1]):
            raise ValueError("expected nonempty query/document matrices with equal embedding dimensions")
        if (not torch.isfinite(query_embeddings).all() or
                not torch.isfinite(document_embeddings).all() or
                (query_embeddings.norm(dim=1) <= 0).any() or
                (document_embeddings.norm(dim=1) <= 0).any()):
            raise ValueError("non-finite or zero-norm embeddings")
        query = F.normalize(query_embeddings, p=2, dim=1)
        documents = F.normalize(document_embeddings, p=2, dim=1)
        scores = self.scale * (query @ documents.T)
        allowed = torch.as_tensor(allowed_mask, device=scores.device)
        if allowed.dtype != torch.bool:
            raise ValueError("allowed_mask must contain booleans")
        labels = torch.as_tensor(targets, device=scores.device)
        if labels.dtype not in (torch.int32, torch.int64):
            raise ValueError("targets must be integer document indices; no float truncation")
        labels = labels.to(dtype=torch.long)
        if allowed.shape != scores.shape or labels.shape != (scores.shape[0],):
            raise ValueError("mask/target shape is incompatible with query/document embeddings")
        if not torch.isfinite(scores).all():
            raise ValueError("non-finite similarity scores")
        if ((labels < 0) | (labels >= scores.shape[1])).any():
            raise ValueError("target document index out of range")
        if not allowed[torch.arange(scores.shape[0], device=scores.device), labels].all():
            raise ValueError("positive document is masked")
        if not (allowed.sum(dim=1) >= 2).all():
            raise ValueError("every row must contain its positive and at least one negative")
        return F.cross_entropy(scores.masked_fill(~allowed, -torch.inf), labels)
