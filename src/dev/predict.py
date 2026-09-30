"""Load a checkpoint and make decisions, with a confidence-gated route."""
import torch
from transformers import AutoTokenizer
from .model import DEV


class Predictor:
    def __init__(self, schema, ckpt_path, threshold=0.85):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.schema, self.threshold = schema, threshold
        self.tok = AutoTokenizer.from_pretrained(ckpt["encoder"])
        self.model = DEV(schema, ckpt["encoder"])
        self.model.load_state_dict(ckpt["state"])
        self.model.eval()

    def decide(self, text, questions=None):
        enc = self.tok([text], truncation=True, max_length=128, padding=True, return_tensors="pt")
        raw = self.model.decide(enc["input_ids"], enc["attention_mask"])
        wanted = questions or list(self.schema)
        decisions = {q: raw[q][0] for q in wanted}
        confs = [d["confidence"] for d in decisions.values() if "confidence" in d]
        route = "auto" if confs and min(confs) >= self.threshold else "review"
        return {"decisions": decisions, "route": route}
