import torch
from accelerate import Accelerator
from prodigyplus.prodigy_plus_schedulefree import ProdigyPlusScheduleFree
from tqdm import tqdm
from transformers import (
    T5EncoderModel,
    T5Tokenizer,
)
from t5_distill.train_utils import LossRecorder, distillation_loss, count_tokens
from t5_distill.config import TrainingConfig
from t5_distill.adapter import ShapeAdapter


class Trainer:
    def __init__(self, config: TrainingConfig) -> None:
        accelerator = Accelerator(log_with="wandb")
        accelerator.init_trackers("t5-distill", config=config)
        generator = torch.Generator("cpu")

        if config.seed is not None:
            generator = generator.manual_seed(config.seed)

        accelerator.print(f"Loading teacher... {config.teacher}")
        teacher = T5EncoderModel.from_pretrained(config.teacher, use_safetensors=True)

        accelerator.print(f"Loading student... {config.student}")
        student = T5EncoderModel.from_pretrained(
            config.student,
            use_safetensors=True,
        )

        if config.teacher_device is not None:
            accelerator.print(f"Moving teacher to device... {config.teacher_device}")
            teacher.to(device=torch.device(config.teacher_device))

        tokenizer = T5Tokenizer.from_pretrained(config.student)

        # Adapter for shape matching
        adapter = ShapeAdapter(
            student.encoder.config.hidden_size, teacher.encoder.config.hidden_size
        )

        accelerator.print(
            f"Adapter in_dim: {student.encoder.config.hidden_size}, out_dim: {teacher.encoder.config.hidden_size}"
        )

        parameters = list(adapter.parameters())

        accelerator.print(f"Adapter parameters: {len(parameters)}")

        accelerator.print("Optimizer: ProdigyPlusScheduleFree")
        accelerator.print(f"Optimizer arguments: {config.optimizer_args}")
        optimizer = ProdigyPlusScheduleFree(parameters, **config.optimizer_args)

        # Tell accelerator if it should auto-place the model on devices
        device_placement = [
            False if config.teacher_device is not None else True,
            False,
            False,
            False,
        ]
        teacher, student, adapter, optimizer = self.accelerator.prepare(
            teacher,
            student,
            adapter,
            optimizer,
            device_placement=device_placement,
        )

        # Models
        self.teacher = teacher
        self.student = student
        self.adapter = adapter

        # Tokenizer
        self.tokenizer = tokenizer

        # Accelerator
        self.accelerator = accelerator

        # Optimizer
        self.optimizer = optimizer

        # Loss function
        self.criterion = distillation_loss

        # Training config
        self.config = config

        # Token counter
        self.__tokens_count = 0

    @property
    def tokens_count(self):
        return self.__tokens_count

    @tokens_count.setter
    def tokens_count(self, count):
        self.__tokens_count = count

    def batch_step(self, batch):
        with self.accelerator.autocast():
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]

            # Get teacher logits
            with torch.no_grad():
                teacher_outputs = self.teacher(
                    input_ids=input_ids.to(self.teacher.device),
                    attention_mask=attention_mask.to(self.teacher.device),
                )
                teacher_logits = teacher_outputs.last_hidden_state

                student_outputs = self.student(
                    input_ids=input_ids, attention_mask=attention_mask
                )
                student_logits = student_outputs.last_hidden_state

            adapted_student_logits = self.adapter(student_logits)

        loss = self.criterion(
            # adapted_student_logits.view(-1, adapted_student_logits.size(-1)).to(
            #     device=teacher_logits.device, dtype=teacher_logits.dtype
            # ),
            # teacher_logits.view(-1, teacher_logits.size(-1)),
            adapted_student_logits.to(
                device=teacher_logits.device, dtype=teacher_logits.dtype
            ),
            teacher_logits,
        )

        return loss

    def train(self, train_dataloader, val_dataloader, epochs):
        self.adapter.train()
        loss_recorder = LossRecorder()
        global_step = 0

        progress_bar = tqdm(
            total=epochs * len(train_dataloader) + epochs * len(val_dataloader),
            smoothing=0,
            disable=not self.accelerator.is_local_main_process,
            desc="steps",
        )

        for epoch in range(epochs):
            self.optimizer.train()
            for step, batch in enumerate(train_dataloader):
                self.optimizer.zero_grad()
                tokens_count = sum(
                    [
                        count_tokens(self.tokenizer, input_ids)
                        for input_ids in batch["input_ids"]
                    ]
                )
                self.tokens_count += tokens_count
                loss = self.batch_step(batch)
                self.accelerator.backward(loss)

                if self.accelerator.sync_gradients:
                    global_step += 1
                    loss_recorder.add(epoch=epoch, step=step, loss=loss.item())

                    progress_bar.update(1)
                    progress_bar.set_postfix(
                        {
                            "epoch": epoch + 1,
                            "lr/d*lr": self.optimizer.param_groups[0]["d"]
                            * self.optimizer.param_groups[0]["lr"],
                            "loss/average": loss_recorder.moving_average,
                        }
                    )
                    self.accelerator.log(
                        {
                            "loss/average": loss_recorder.moving_average,
                            "tokens_count": tokens_count,
                            "lr/d*lr": self.optimizer.param_groups[0]["d"]
                            * self.optimizer.param_groups[0]["lr"],
                        },
                        step=global_step,
                    )

                self.optimizer.step()

                if self.config.steps is not None:
                    if global_step > self.config.steps:
                        break

            if self.config.steps is not None:
                if global_step > self.config.steps:
                    break

            self.optimizer.eval()
            val_loss_recorder = LossRecorder()
            for step, batch in enumerate(val_dataloader):
                global_step += 1
                loss = self.batch_step(batch)
                tokens_count = sum(
                    [
                        count_tokens(self.tokenizer, input_ids)
                        for input_ids in batch["input_ids"]
                    ]
                )
                val_loss_recorder.add(epoch=epoch, step=step, loss=loss.item())

                progress_bar.update(1)
                progress_bar.set_postfix(
                    {
                        "epoch": epoch + 1,
                        "lr/d*lr": self.optimizer.param_groups[0]["d"]
                        * self.optimizer.param_groups[0]["lr"],
                        "val_loss/average": val_loss_recorder.moving_average,
                        "loss/average": loss_recorder.moving_average,
                    }
                )
                # print(f"Epoch {epoch + 1}, Avg: {loss_recorder.moving_average:.4f}")
                self.accelerator.log(
                    {
                        "val_step": step,
                        "tokens_count": tokens_count,
                        "val_loss/average": val_loss_recorder.moving_average,
                        "lr/d*lr": self.optimizer.param_groups[0]["d"]
                        * self.optimizer.param_groups[0]["lr"],
                    },
                    step=global_step,
                )
