import torch
from typing import List
from transformers import T5Tokenizer
from torch import nn
from torch import Tensor
import torch.nn.functional as F


def distillation_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    student_temperature=2.0,
    teacher_temperature=2.0,
):
    return (
        nn.KLDivLoss(reduction="batchmean")(
            F.log_softmax(student_logits / student_temperature, dim=-1),
            F.softmax(teacher_logits / teacher_temperature, dim=-1),
        )
        * student_temperature
        * teacher_temperature
    )


def count_tokens(tokenizer: T5Tokenizer, input_ids: torch.Tensor):
    # Filter out padding and special tokens
    filtered_tokens = [
        id
        for id in input_ids
        if id.item() not in tokenizer.all_special_ids
        and id.item() != tokenizer.pad_token_id
    ]

    return len(filtered_tokens)


# https://github.com/kohya-ss/sd-scripts/blob/main/library/train_util.py
class LossRecorder:
    def __init__(self):
        self.loss_list: List[float] = []
        self.loss_total: float = 0.0

    def add(self, *, epoch: int, step: int, loss: float) -> None:
        if epoch == 0:
            self.loss_list.append(loss)
        else:
            while len(self.loss_list) <= step:
                self.loss_list.append(0.0)
            self.loss_total -= self.loss_list[step]
            self.loss_list[step] = loss
        self.loss_total += loss

    @property
    def moving_average(self) -> float:
        losses = len(self.loss_list)
        if losses == 0:
            return 0

        return self.loss_total / losses
