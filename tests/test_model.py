import pytest
torch = pytest.importorskip("torch")
from transformers import BertConfig
from dev.schema import load_schema
from dev.model import DEV


def tiny_model():
    cfg = BertConfig(hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                     intermediate_size=64, vocab_size=100)
    return DEV(load_schema("configs/schema.yaml"), encoder_config=cfg)


def test_forward_shapes():
    m = tiny_model()
    ids = torch.randint(0, 100, (3, 10)); mask = torch.ones(3, 10, dtype=torch.long)
    out = m(ids, mask)
    assert out["department"].shape == (3, 4) and out["urgency"].shape == (3, 1)


def test_loss_backward_and_decide():
    m = tiny_model()
    ids = torch.randint(0, 100, (3, 10)); mask = torch.ones(3, 10, dtype=torch.long)
    labels = {"department": torch.tensor([0, 1, 2]), "urgency": torch.tensor([0.1, 0.5, 0.9]),
              "escalate": torch.tensor([0., 1., 1.])}
    loss = m.loss(m(ids, mask), labels)
    loss.backward()
    assert loss.item() > 0
    d = m.decide(ids, mask)
    assert abs(sum(d["department"][0]["probs"].values()) - 1) < 1e-5
    assert isinstance(d["escalate"][0]["value"], bool)
