import pytest
import torch

from trl.experimental.gold.gold_config import GOLDConfig
from trl.experimental.gold.gold_trainer import GOLDTrainer


def test_gold_config_reverse_kl_topk_defaults():
    config = GOLDConfig(output_dir="tmp")
    assert config.distill_loss_type == "jsd"
    assert config.reverse_kl_topk == 1


def test_gold_config_invalid_distill_loss_type():
    with pytest.raises(ValueError, match="distill_loss_type must be one of"):
        GOLDConfig(output_dir="tmp", distill_loss_type="unknown")


def test_gold_config_invalid_reverse_kl_topk():
    with pytest.raises(ValueError, match="reverse_kl_topk must be >= 1"):
        GOLDConfig(output_dir="tmp", reverse_kl_topk=0)


def test_gold_config_reverse_kl_topk_requires_on_policy():
    with pytest.raises(ValueError, match="requires lmbda=1.0"):
        GOLDConfig(output_dir="tmp", distill_loss_type="reverse_kl_topk", lmbda=0.5)


def test_gold_config_reverse_kl_topk_disallows_uld():
    with pytest.raises(ValueError, match="does not support use_uld_loss=True"):
        GOLDConfig(output_dir="tmp", distill_loss_type="reverse_kl_topk", lmbda=1.0, use_uld_loss=True)


def test_gold_config_reverse_kl_topk_disallows_liger():
    with pytest.raises(ValueError, match="does not support use_liger_kernel=True"):
        GOLDConfig(output_dir="tmp", distill_loss_type="reverse_kl_topk", lmbda=1.0, use_liger_kernel=True)


def test_reverse_kl_topk_loss_top1_uses_rollout_token_and_student_prob_weight():
    student_logits = torch.tensor([[[2.0, 0.0, -1.0], [0.1, -0.2, 0.3]]], dtype=torch.float32)
    teacher_logits = torch.tensor([[[1.0, 0.5, -0.5], [0.0, 0.0, 0.0]]], dtype=torch.float32)
    labels = torch.tensor([[1, -100]], dtype=torch.long)

    loss = GOLDTrainer.reverse_kl_topk_loss(student_logits, teacher_logits, labels, topk=1)

    student_log_probs = torch.log_softmax(student_logits[0, 0], dim=-1)
    teacher_log_probs = torch.log_softmax(teacher_logits[0, 0], dim=-1)
    rollout_token = labels[0, 0].item()
    p_s = torch.exp(student_log_probs[rollout_token])
    expected = p_s * (student_log_probs[rollout_token] - teacher_log_probs[rollout_token])

    torch.testing.assert_close(loss, expected)


def test_reverse_kl_topk_loss_topk_truncates_student_distribution():
    student_logits = torch.tensor([[[3.0, 2.0, 0.0, -1.0]]], dtype=torch.float32)
    teacher_logits = torch.tensor([[[2.5, 1.0, 0.5, -2.0]]], dtype=torch.float32)
    labels = torch.tensor([[0]], dtype=torch.long)

    loss = GOLDTrainer.reverse_kl_topk_loss(student_logits, teacher_logits, labels, topk=2)

    s_log = torch.log_softmax(student_logits[0, 0], dim=-1)
    t_log = torch.log_softmax(teacher_logits[0, 0], dim=-1)
    s_prob = torch.softmax(student_logits[0, 0], dim=-1)
    topk_indices = torch.topk(s_prob, k=2, dim=-1).indices
    expected = (s_prob[topk_indices] * (s_log[topk_indices] - t_log[topk_indices])).sum()

    torch.testing.assert_close(loss, expected)


def test_reverse_kl_topk_loss_masks_ignore_index():
    student_logits = torch.randn(1, 2, 4)
    teacher_logits = torch.randn(1, 2, 4)
    labels = torch.full((1, 2), -100, dtype=torch.long)

    loss = GOLDTrainer.reverse_kl_topk_loss(student_logits, teacher_logits, labels, topk=1)
    assert loss.item() == 0.0
