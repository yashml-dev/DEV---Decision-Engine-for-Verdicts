"""Train DEV:  python -m dev.train --epochs 3"""
import argparse
import random
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from .schema import load_schema
from .model import DEV
from .data import make_synthetic, save_jsonl, load_jsonl, TicketDataset


def evaluate(model, loader, schema, device):
    model.eval()
    correct = {q: 0 for q, s in schema.items() if s.type != "score"}
    total = 0
    with torch.no_grad():
        for ids, mask, labels in loader:
            logits = model(ids.to(device), mask.to(device))
            total += ids.size(0)
            for qid, q in schema.items():
                if q.type == "choice":
                    correct[qid] += (logits[qid].argmax(-1).cpu() == labels[qid]).sum().item()
                elif q.type == "yes_no":
                    correct[qid] += ((logits[qid].squeeze(-1).cpu() > 0) == (labels[qid] > 0.5)).sum().item()
    return {k: v / total for k, v in correct.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--schema", default="configs/schema.yaml")
    p.add_argument("--data", default="data/samples/tickets.jsonl")
    p.add_argument("--encoder", default="microsoft/MiniLM-L12-H384-uncased")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--out", default="checkpoints/dev-v0.1.pt")
    a = p.parse_args()

    random.seed(0); torch.manual_seed(0)
    schema = load_schema(a.schema)
    try:
        rows = load_jsonl(a.data)
    except FileNotFoundError:
        print("No data found - generating synthetic tickets.")
        rows = make_synthetic(600)
        save_jsonl(rows, a.data)
    random.shuffle(rows)
    cut = int(0.8 * len(rows))
    tok = AutoTokenizer.from_pretrained(a.encoder)
    train_ds, val_ds = TicketDataset(rows[:cut], schema, tok), TicketDataset(rows[cut:], schema, tok)
    train_dl = DataLoader(train_ds, a.batch_size, shuffle=True, collate_fn=train_ds.collate)
    val_dl = DataLoader(val_ds, a.batch_size, collate_fn=val_ds.collate)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DEV(schema, a.encoder).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)

    for epoch in range(a.epochs):
        model.train()
        for ids, mask, labels in train_dl:
            labels = {k: v.to(device) for k, v in labels.items()}
            loss = model.loss(model(ids.to(device), mask.to(device)), labels)
            opt.zero_grad(); loss.backward(); opt.step()
        print(f"epoch {epoch + 1}: loss={loss.item():.4f}  val_acc={evaluate(model, val_dl, schema, device)}")

    import os
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    torch.save({"state": model.state_dict(), "encoder": a.encoder}, a.out)
    print(f"Saved {a.out}")


if __name__ == "__main__":
    main()
