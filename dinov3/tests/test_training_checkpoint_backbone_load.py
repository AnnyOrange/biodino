import torch

from dinov3.checkpointer.checkpointer import init_fsdp_model_from_checkpoint


def test_selects_teacher_backbone_from_training_checkpoint(tmp_path):
    target = torch.nn.ModuleDict({"backbone": torch.nn.Linear(3, 2, bias=False)})
    teacher_weight = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    checkpoint = {
        "iteration": 10,
        "model": {
            "teacher.backbone.weight": teacher_weight,
            "student.backbone.weight": torch.zeros_like(teacher_weight),
        },
    }
    checkpoint_path = tmp_path / "checkpoint.pth"
    torch.save(checkpoint, checkpoint_path)

    init_fsdp_model_from_checkpoint(
        target,
        str(checkpoint_path),
        checkpoint_state_prefix="teacher.backbone.",
    )

    assert torch.equal(target["backbone"].weight, teacher_weight)
