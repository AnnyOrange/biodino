import pytest
import torch

from dinov3.loss import gram_loss as gram_loss_module
from dinov3.loss.gram_loss import CrossRankGramLoss, GramLoss


def test_gram_loss_supports_per_crop_inter_image_relations():
    teacher = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            [[1.0, 1.0], [1.0, -1.0], [0.0, 1.0]],
        ]
    )
    student = teacher.clone().requires_grad_(True)

    loss = GramLoss(apply_norm=True, remove_neg=False)(student, teacher, img_level=True)

    assert loss.item() == pytest.approx(0.0, abs=1e-8)
    loss.backward()
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()


def test_inter_image_relation_gram_detects_changed_pairwise_geometry():
    teacher = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
    student = teacher.clone()
    student[0, 2] = torch.tensor([-1.0, 1.0])

    loss = GramLoss(apply_norm=True, remove_neg=False)(student, teacher, img_level=True)

    assert loss.item() > 0.1


def test_cross_rank_gram_loss_matches_relations_and_stops_teacher_grad():
    teacher = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]], requires_grad=True
    )
    student = teacher.detach().clone().requires_grad_(True)
    loss_fn = CrossRankGramLoss(relation_scope="global")

    loss = loss_fn(student, teacher)

    assert loss.item() == pytest.approx(0.0, abs=1e-8)
    loss.backward()
    assert student.grad is not None
    assert teacher.grad is None


def test_cross_rank_gram_gathers_each_view_along_the_batch_axis(monkeypatch):
    local = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    loss_fn = CrossRankGramLoss(relation_scope="global")

    monkeypatch.setattr(gram_loss_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(gram_loss_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(gram_loss_module.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(
        gram_loss_module.dist_nn,
        "all_gather",
        lambda value: (value, value + 100),
    )

    gathered = loss_fn._gather_views(local, with_grad=True)

    assert gathered.shape == (2, 6, 4)
    torch.testing.assert_close(gathered[:, :3], local)
    torch.testing.assert_close(gathered[:, 3:], local + 100)
