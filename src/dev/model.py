"""The DEV model: one shared encoder + one typed head per question."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, PretrainedConfig


class DEV(nn.Module):
    def __init__(self, schema, encoder_name="microsoft/MiniLM-L12-H384-uncased",
                 encoder_config: PretrainedConfig = None):
        super().__init__()
        self.schema = schema
        if encoder_config is not None:      # used by tests: tiny random encoder, no download
            self.encoder = AutoModel.from_config(encoder_config)
        else:
            self.encoder = AutoModel.from_pretrained(encoder_name)
        h = self.encoder.config.hidden_size
        self.heads = nn.ModuleDict()
        for qid, q in schema.items():
            out_dim = len(q.options) if q.type == "choice" else 1
            self.heads[qid] = nn.Linear(h, out_dim)

    def _pool(self, hidden, mask):
        """Mean-pool token vectors into one summary vector per text."""
        m = mask.unsqueeze(-1).float()
        return (hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)

    def forward(self, input_ids, attention_mask):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        summary = self._pool(hidden, attention_mask)
        return {qid: head(summary) for qid, head in self.heads.items()}  # raw logits

    def loss(self, logits, labels):
        """Multi-task loss. labels[qid]: class index (choice) / float in [0,1] (score) / 0-1 (yes_no)."""
        total = 0.0
        for qid, q in self.schema.items():
            if q.type == "choice":
                total = total + F.cross_entropy(logits[qid], labels[qid].long())
            elif q.type == "score":
                total = total + F.mse_loss(logits[qid].squeeze(-1).sigmoid(), labels[qid].float())
            else:
                total = total + F.binary_cross_entropy_with_logits(
                    logits[qid].squeeze(-1), labels[qid].float())
        return total

    @torch.no_grad()
    def decide(self, input_ids, attention_mask):
        """Turn logits into typed decisions with probabilities and confidence."""
        self.eval()
        logits = self.forward(input_ids, attention_mask)
        out = {}
        for qid, q in self.schema.items():
            if q.type == "choice":
                probs = logits[qid].softmax(-1)
                conf, idx = probs.max(-1)
                out[qid] = [
                    {"value": q.options[i], "confidence": float(c),
                     "probs": {o: float(p) for o, p in zip(q.options, row)}}
                    for i, c, row in zip(idx.tolist(), conf.tolist(), probs)
                ]
            elif q.type == "score":
                out[qid] = [{"value": float(v)} for v in logits[qid].squeeze(-1).sigmoid()]
            else:
                p = logits[qid].squeeze(-1).sigmoid()
                out[qid] = [{"value": bool(v > 0.5), "prob": float(v),
                             "confidence": float(max(v, 1 - v))} for v in p]
        return out
