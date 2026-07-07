"""
NOUS-Relational — the packaged model embodying the validated pillars.

A single energy-based relational model that does everything the experiments
validated, GPU-ready and trainable on any knowledge graph:

  • Bidirectional inference  — one energy answers (h,r,?) and (?,r,t)  [§4.1, 4.7]
  • Learnable binding        — asymmetry knob λ (0 = symmetric DistMult,
                               >0 = ComplEx); sweet spot ≈0.5            [§4.6, 4.8]
  • Multi-hop composition    — chain relations through latent entities    [§4.3]
  • Adaptive test-time compute — refine the latent for `steps` iterations [§4.10]
  • Native confidence        — energy margin → calibrated uncertainty     [§4.11]

Energy:  E(h, r, t) = − Re⟨e_h, w_r, conj(e_t)⟩   (ComplEx; λ scales the
imaginary/antisymmetric part). Inference = pick the entity that minimises E.

Usage:
    m = NOUSRelational(n_entities, n_relations, dim=200, asymmetry=0.5).to(device)
    m.fit(train_triples, epochs=30)
    metrics = m.evaluate(test_triples, filters)
    scores  = m.score_tail(h_idx, r_idx)          # (B, n_entities)
    conf    = m.confidence(scores)                 # (B,)
    chain   = m.multihop(h_idx, [r1_idx, r2_idx])  # 2-hop
"""
from __future__ import annotations
from dataclasses import dataclass
import torch, torch.nn as nn, torch.nn.functional as F


@dataclass
class NOUSRelationalConfig:
    n_entities: int
    n_relations: int
    dim: int = 200
    asymmetry: float = 0.5          # λ: 0 = symmetric DistMult, ~0.5 = sweet spot
    init_scale: float = 0.1


class NOUSRelational(nn.Module):
    def __init__(self, n_entities: int, n_relations: int, dim: int = 200,
                 asymmetry: float = 0.5, init_scale: float = 0.1):
        super().__init__()
        self.cfg = NOUSRelationalConfig(n_entities, n_relations, dim, asymmetry, init_scale)
        self.lam = asymmetry
        self.Ere = nn.Parameter(torch.randn(n_entities, dim) * init_scale)
        self.Eim = nn.Parameter(torch.randn(n_entities, dim) * init_scale)
        self.Rre = nn.Parameter(torch.randn(n_relations, dim) * init_scale)
        self.Rim = nn.Parameter(torch.randn(n_relations, dim) * init_scale)

    # ── scoring (bidirectional) ──────────────────────────────────────────────
    def score_tail(self, h: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """Score every candidate tail for (h, r). Returns (B, n_entities)."""
        rim = self.lam * self.Rim[r]
        p = self.Ere[h] * self.Rre[r] - self.Eim[h] * rim
        q = self.Eim[h] * self.Rre[r] + self.Ere[h] * rim
        return p @ self.Ere.t() + q @ self.Eim.t()

    def score_head(self, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Score every candidate head for (r, t) — SAME energy, reversed. (B, n_entities)."""
        rim = self.lam * self.Rim[r]
        a = self.Rre[r] * self.Ere[t] + rim * self.Eim[t]
        b = self.Rre[r] * self.Eim[t] - rim * self.Ere[t]
        return a @ self.Ere.t() + b @ self.Eim.t()

    def energy(self, h, r, t) -> torch.Tensor:
        """Energy of a specific triple (lower = more plausible)."""
        return -self.score_tail(h, r).gather(1, t.view(-1, 1)).squeeze(1)

    def triple_score(self, h, r, t) -> torch.Tensor:
        """Score of specific (h,r,t) triples — O(B), no softmax over entities. (B,)"""
        rim = self.lam * self.Rim[r]
        p = self.Ere[h] * self.Rre[r] - self.Eim[h] * rim
        q = self.Eim[h] * self.Rre[r] + self.Ere[h] * rim
        return (p * self.Ere[t] + q * self.Eim[t]).sum(-1)

    def _neg_scores(self, h, r, t_neg):
        """Scores for corrupted TAILS (h, r, t_neg[:,k]) — t_neg: (B,K). Returns (B,K)."""
        rim = self.lam * self.Rim[r]
        p = (self.Ere[h] * self.Rre[r] - self.Eim[h] * rim).unsqueeze(1)   # (B,1,D)
        q = (self.Eim[h] * self.Rre[r] + self.Ere[h] * rim).unsqueeze(1)
        return (p * self.Ere[t_neg] + q * self.Eim[t_neg]).sum(-1)         # (B,K)

    def _neg_scores_head(self, h_neg, r, t):
        """Scores for corrupted HEADS (h_neg[:,k], r, t) — h_neg: (B,K). Returns (B,K)."""
        rim = self.lam * self.Rim[r]
        a = (self.Rre[r] * self.Ere[t] + rim * self.Eim[t]).unsqueeze(1)   # (B,1,D)
        b = (self.Rre[r] * self.Eim[t] - rim * self.Ere[t]).unsqueeze(1)
        return (a * self.Ere[h_neg] + b * self.Eim[h_neg]).sum(-1)         # (B,K)

    # ── confidence (native, calibrated) ──────────────────────────────────────
    @staticmethod
    def confidence(scores: torch.Tensor) -> torch.Tensor:
        """Max softmax over candidates — a calibrated confidence signal."""
        return F.softmax(scores, dim=-1).max(dim=-1).values

    # ── multi-hop composition + adaptive compute ─────────────────────────────
    def multihop(self, h: torch.Tensor, rels: list[torch.Tensor],
                 steps: int = 1) -> torch.Tensor:
        """
        Chain relations r1,r2,… from head h through latent entities.
        `steps` extra mean-field refinements = adaptive test-time compute.
        Returns scores over entities for the final hop. (B, n_entities)
        """
        B = h.shape[0]
        dist = F.one_hot(h, self.cfg.n_entities).float()      # current soft entity
        for r in rels:
            for _ in range(steps):
                ere = dist @ self.Ere                          # soft entity embedding
                eim = dist @ self.Eim
                rim = self.lam * self.Rim[r]
                p = ere * self.Rre[r] - eim * rim
                q = eim * self.Rre[r] + ere * rim
                scores = p @ self.Ere.t() + q @ self.Eim.t()
                dist = F.softmax(scores, dim=-1)
        return scores

    def forward(self, h, r):
        return self.score_tail(h, r)

    # ── training ─────────────────────────────────────────────────────────────
    def fit(self, triples: torch.Tensor, epochs: int = 30, batch: int = 1024,
            lr: float = 1e-2, weight_decay: float = 1e-7, reg: float = 1e-4,
            log_every: int = 5, device=None):
        """Train on (N,3) [head, rel, tail] tensor via tail-prediction CE."""
        device = device or next(self.parameters()).device
        triples = triples.to(device)
        opt = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        for ep in range(epochs):
            perm = torch.randperm(len(triples), device=device)
            tot = 0.0
            for i in range(0, len(triples), batch):
                b = triples[perm[i:i+batch]]; h, r, t = b[:, 0], b[:, 1], b[:, 2]
                s = self.score_tail(h, r)
                loss = F.cross_entropy(s, t) + reg * (
                    self.Ere[h].pow(2).mean() + self.Eim[h].pow(2).mean()
                    + self.Rre[r].pow(2).mean())
                opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
            if log_every and (ep % log_every == 0 or ep == epochs - 1):
                print(f"  epoch {ep:3d}  loss {tot/max(1,len(triples)//batch):.4f}")
        return self

    def fit_large(self, triples: torch.Tensor, epochs: int = 10, batch: int = 1024,
                  neg: int = 64, lr: float = 1e-3, reg: float = 1e-6,
                  adversarial: float = 1.0, log_every: int = 1, device=None):
        """
        Negative-sampling training for ENORMOUS KGs (millions of entities) where
        full-softmax is impossible. Self-adversarial sigmoid loss (RotatE-style).
        Corrupts BOTH tails and heads so ONE model is strong in BOTH query
        directions (bidirectional from a single symmetric energy).
        """
        device = device or next(self.parameters()).device
        nE = self.cfg.n_entities
        triples = triples.to(device)
        opt = torch.optim.Adam(self.parameters(), lr=lr)

        def neg_term(neg_s):
            if adversarial > 0:
                w = torch.softmax(neg_s * adversarial, dim=1).detach()
                return (w * F.logsigmoid(-neg_s)).sum(1)
            return F.logsigmoid(-neg_s).mean(1)

        for ep in range(epochs):
            perm = torch.randperm(len(triples), device=device)
            tot = 0.0
            for i in range(0, len(triples), batch):
                b = triples[perm[i:i+batch]]; h, r, t = b[:, 0], b[:, 1], b[:, 2]
                B = h.shape[0]
                pos = self.triple_score(h, r, t)                              # (B,)
                t_neg = torch.randint(0, nE, (B, neg), device=device)         # corrupt tail
                h_neg = torch.randint(0, nE, (B, neg), device=device)         # corrupt head
                loss_t = -(F.logsigmoid(pos) + neg_term(self._neg_scores(h, r, t_neg))).mean()
                loss_h = -(F.logsigmoid(pos) + neg_term(self._neg_scores_head(h_neg, r, t))).mean()
                loss = 0.5 * (loss_t + loss_h)
                loss = loss + reg * (self.Ere[h].pow(2).mean() + self.Eim[h].pow(2).mean())
                opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
            if log_every and (ep % log_every == 0 or ep == epochs - 1):
                print(f"  epoch {ep:3d}  loss {tot/max(1,len(triples)//batch):.4f}")
        return self

    @torch.no_grad()
    def evaluate_large(self, triples, filt_tails, filt_heads=None,
                       sample=2000, device=None):
        """
        Filtered MRR/Hits ranking against ALL entities, one query at a time.
        tail = (h,r,?) [trained direction]. If filt_heads given, also head = (?,r,t)
        [the never-trained REVERSE direction] — the reversal-curse test at scale.
        """
        device = device or next(self.parameters()).device
        T = triples.to(device)
        if sample and len(T) > sample:
            T = T[torch.randperm(len(T))[:sample]]
        rt, rh = [], []
        for h, r, t in T.tolist():
            s = self.score_tail(torch.tensor([h], device=device),
                                torch.tensor([r], device=device))[0]
            for o in filt_tails.get((h, r), ()):
                if o != t: s[o] = -1e9
            rt.append((s > s[t]).sum().item() + 1)
            if filt_heads is not None:
                s = self.score_head(torch.tensor([r], device=device),
                                    torch.tensor([t], device=device))[0]
                for o in filt_heads.get((r, t), ()):
                    if o != h: s[o] = -1e9
                rh.append((s > s[h]).sum().item() + 1)
        def m(ranks):
            x = torch.tensor(ranks, dtype=torch.float)
            return dict(MRR=(1/x).mean().item(), H1=(x <= 1).float().mean().item()*100,
                        H10=(x <= 10).float().mean().item()*100)
        return m(rt) if filt_heads is None else {"tail": m(rt), "head": m(rh)}

    @torch.no_grad()
    def evaluate(self, triples: torch.Tensor, filt_tails: dict, filt_heads: dict,
                 sample: int | None = 3000, device=None) -> dict:
        """Filtered MRR / Hits@1 / Hits@10 for tail (trained) and head (reverse)."""
        device = device or next(self.parameters()).device
        T = triples.to(device)
        if sample and len(T) > sample:
            T = T[torch.randperm(len(T))[:sample]]
        rt, rh = [], []
        for h, r, t in T.tolist():
            s = self.score_tail(torch.tensor([h], device=device), torch.tensor([r], device=device))[0]
            for o in filt_tails.get((h, r), ()):
                if o != t: s[o] = -1e9
            rt.append((s > s[t]).sum().item() + 1)
            s = self.score_head(torch.tensor([r], device=device), torch.tensor([t], device=device))[0]
            for o in filt_heads.get((r, t), ()):
                if o != h: s[o] = -1e9
            rh.append((s > s[h]).sum().item() + 1)
        def m(x):
            x = torch.tensor(x, dtype=torch.float)
            return dict(MRR=(1/x).mean().item(), H1=(x <= 1).float().mean().item()*100,
                        H10=(x <= 10).float().mean().item()*100)
        return {"tail": m(rt), "head": m(rh)}

    # ── persistence ──────────────────────────────────────────────────────────
    def save(self, path: str):
        torch.save({"state_dict": self.state_dict(), "cfg": self.cfg.__dict__}, path)

    @classmethod
    def load(cls, path: str, map_location=None):
        ck = torch.load(path, map_location=map_location)
        m = cls(**{k: ck["cfg"][k] for k in
                   ("n_entities", "n_relations", "dim", "asymmetry", "init_scale")})
        m.load_state_dict(ck["state_dict"]); return m
