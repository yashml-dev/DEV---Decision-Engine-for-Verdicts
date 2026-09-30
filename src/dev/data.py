"""Synthetic ticket generator (for a first end-to-end run) + PyTorch dataset."""
import json
import random
from pathlib import Path

TEMPLATES = {
    "billing": ["I was charged twice this month", "My invoice shows the wrong amount",
                "Please refund my last payment", "Why was my card billed again?"],
    "technical": ["The app crashes when I open settings", "I get an error 500 on login page",
                  "The website is very slow and freezes", "Sync is not working on my phone"],
    "account": ["I cannot reset my password", "Please change the email on my account",
                "My account was locked", "How do I delete my profile?"],
    "general": ["What are your support hours?", "Do you have a student discount?",
                "Where can I read your terms?", "I have a question about your plans"],
}
URGENT = ["This is urgent!", "Need help immediately.", "Production is down.", "ASAP please."]
CALM = ["No rush.", "Whenever you can.", "Just curious.", ""]


def make_synthetic(n=600, seed=0):
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        dept = rng.choice(list(TEMPLATES))
        urgent = rng.random() < 0.3
        text = f"{rng.choice(TEMPLATES[dept])} {rng.choice(URGENT if urgent else CALM)}".strip()
        rows.append({"text": text, "department": dept,
                     "urgency": round(rng.uniform(0.7, 1.0) if urgent else rng.uniform(0.0, 0.3), 2),
                     "escalate": int(urgent and dept in ("billing", "technical"))})
    return rows


def save_jsonl(rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(json.dumps(r) for r in rows))


def load_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


class TicketDataset:
    """Tokenizes text and converts labels according to the schema."""
    def __init__(self, rows, schema, tokenizer, max_len=128):
        self.rows, self.schema, self.tok, self.max_len = rows, schema, tokenizer, max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]

    def collate(self, batch):
        import torch
        enc = self.tok([r["text"] for r in batch], padding=True, truncation=True,
                       max_length=self.max_len, return_tensors="pt")
        labels = {}
        for qid, q in self.schema.items():
            if q.type == "choice":
                labels[qid] = torch.tensor([q.options.index(r[qid]) for r in batch])
            else:
                labels[qid] = torch.tensor([float(r[qid]) for r in batch])
        return enc["input_ids"], enc["attention_mask"], labels
