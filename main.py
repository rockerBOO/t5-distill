import argparse
import ast
import datetime

import torch
from accelerate import Accelerator
from prodigyplus.prodigy_plus_schedulefree import ProdigyPlusScheduleFree
from safetensors.torch import save_file
from tqdm import tqdm
from transformers import (
    T5EncoderModel,
    T5Tokenizer,
)
from t5_distill.dataloader import create_boolq_dataloaders
from t5_distill.train_utils import LossRecorder, distillation_loss, count_tokens
from t5_distill.config import TrainingConfig, compile_metadata
from t5_distill.adapter import ShapeAdapter


class Trainer:
    def __init__(self, config: TrainingConfig) -> None:
        accelerator = Accelerator(log_with="wandb")
        accelerator.init_trackers("t5-distill", config=config)
        generator = torch.Generator("cpu")

        if config.seed is not None:
            generator = generator.manual_seed(config.seed)

        accelerator.print(f"Loading teacher... {args.teacher}")
        teacher = T5EncoderModel.from_pretrained(args.teacher, use_safetensors=True)

        accelerator.print(f"Loading student... {args.student}")
        student = T5EncoderModel.from_pretrained(
            args.student,
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
        accelerator.print(f"Optimizer arguments: {args.optimizer_args}")
        optimizer = ProdigyPlusScheduleFree(parameters, **args.optimizer_args)

        # Tell accelerator if it should auto-place the model on devices
        device_placement = [
            False if args.teacher_device is not None else True,
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

                if args.steps is not None:
                    if global_step > args.steps:
                        break

            if args.steps is not None:
                if global_step > args.steps:
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


def main(args: TrainingConfig):
    trainer = Trainer(args)

    train_loader, val_loader = create_boolq_dataloaders(
        trainer.tokenizer,
        batch_size=args.batch_size,
        test_size=args.validation_split if args.validation_split is not None else 0,
        cache_dir=args.dataset_cache_dir,
    )

    train_loader, val_loader = trainer.accelerator.prepare(
        train_loader,
        val_loader,
    )

    trainer.train(train_loader, val_loader, epochs=args.epochs)

    trainer.accelerator.end_training()

    end_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    metadata = compile_metadata(args)
    metadata["end_timestamp"] = end_timestamp
    metadata["tokens_count"] = str(trainer.tokens_count)
    trainer.accelerator.print(f"Total tokens count: {metadata['tokens_count']}")
    save_file(
        trainer.adapter.state_dict(),
        f"adapter-t5-efficient-small-{end_timestamp}.safetensors",
        metadata=metadata,
    )

    # accelerator = Accelerator(log_with="wandb")
    # accelerator.init_trackers("t5-distill", config=args)
    # generator = torch.Generator("cpu")
    #
    # if args.seed is not None:
    #     generator = generator.manual_seed(args.seed)
    #
    # accelerator.print(f"Loading teacher... {args.teacher}")
    # teacher = T5EncoderModel.from_pretrained(args.teacher, use_safetensors=True)
    #
    # accelerator.print(f"Loading student... {args.student}")
    # student = T5EncoderModel.from_pretrained(
    #     args.student,
    #     use_safetensors=True,
    # )
    #
    # if args.teacher_device is not None:
    #     accelerator.print(f"Moving teacher to device... {args.teacher_device}")
    #     teacher.to(device=torch.device(args.teacher_device))
    #
    # tokenizer = T5Tokenizer.from_pretrained(args.student)
    #
    # # Adapter for shape matching
    # adapter = ShapeAdapter(
    #     student.encoder.config.hidden_size, teacher.encoder.config.hidden_size
    # )
    #
    # accelerator.print(
    #     f"Adapter in_dim: {student.encoder.config.hidden_size}, out_dim: {teacher.encoder.config.hidden_size}"
    # )
    #
    # parameters = list(adapter.parameters())
    #
    # accelerator.print(f"Adapter parameters: {len(parameters)}")
    #
    # accelerator.print("Optimizer: ProdigyPlusScheduleFree")
    # accelerator.print(f"Optimizer arguments: {args.optimizer_args}")
    # optimizer = ProdigyPlusScheduleFree(parameters, **args.optimizer_args)
    #
    # criterion = distillation_loss
    #
    # @accelerator.autocast()
    # def train_step(batch: dict[str, torch.Tensor]):
    #     input_ids = batch["input_ids"]
    #     attention_mask = batch["attention_mask"]
    #
    #     # Get teacher logits
    #     with torch.no_grad():
    #         teacher_outputs = teacher(
    #             input_ids=input_ids.to(teacher.device),
    #             attention_mask=attention_mask.to(teacher.device),
    #         )
    #         teacher_logits = teacher_outputs.last_hidden_state
    #
    #         student_outputs = student(
    #             input_ids=input_ids, attention_mask=attention_mask
    #         )
    #         student_logits = student_outputs.last_hidden_state
    #
    #     adapted_student_logits = adapter(student_logits)
    #
    #     loss = criterion(
    #         # adapted_student_logits.view(-1, adapted_student_logits.size(-1)).to(
    #         #     device=teacher_logits.device, dtype=teacher_logits.dtype
    #         # ),
    #         # teacher_logits.view(-1, teacher_logits.size(-1)),
    #         adapted_student_logits.to(
    #             device=teacher_logits.device, dtype=teacher_logits.dtype
    #         ),
    #         teacher_logits,
    #     )
    #
    #     return loss
    #
    # @accelerator.autocast()
    # def train(train_dataloader, val_dataloader, epochs):
    #     adapter.train()
    #     loss_recorder = LossRecorder()
    #     global_step = 0
    #
    #     progress_bar = tqdm(
    #         total=epochs * len(train_dataloader) + epochs * len(val_dataloader),
    #         smoothing=0,
    #         disable=not accelerator.is_local_main_process,
    #         desc="steps",
    #     )
    #
    #     for epoch in range(epochs):
    #         optimizer.train()
    #         for step, batch in enumerate(train_dataloader):
    #             optimizer.zero_grad()
    #             tokens_count = sum(
    #                 [
    #                     count_tokens(tokenizer, input_ids)
    #                     for input_ids in batch["input_ids"], asdict
    #                 ]
    #             )
    #             accumulate_tokens_count(tokens_count)
    #             loss = train_step(batch)
    #             accelerator.backward(loss)
    #
    #             if accelerator.sync_gradients:
    #                 global_step += 1
    #                 loss_recorder.add(epoch=epoch, step=step, loss=loss.item())
    #
    #                 progress_bar.update(1)
    #                 progress_bar.set_postfix(
    #                     {
    #                         "epoch": epoch + 1,
    #                         "lr/d*lr": optimizer.param_groups[0]["d"]
    #                         * optimizer.param_groups[0]["lr"],
    #                         "loss/average": loss_recorder.moving_average,
    #                     }
    #                 )
    #                 accelerator.log(
    #                     {
    #                         "loss/average": loss_recorder.moving_average,
    #                         "tokens_count": tokens_count,
    #                         "lr/d*lr": optimizer.param_groups[0]["d"]
    #                         * optimizer.param_groups[0]["lr"],
    #                     },
    #                     step=global_step,
    #                 )
    #
    #             optimizer.step()
    #
    #             if args.steps is not None:
    #                 if global_step > args.steps:
    #                     break
    #
    #         if args.steps is not None:
    #             if global_step > args.steps:
    #                 break
    #
    #         optimizer.eval()
    #         val_loss_recorder = LossRecorder()
    #         for step, batch in enumerate(val_dataloader):
    #             global_step += 1
    #             loss = train_step(batch)
    #             tokens_count = sum(
    #                 [
    #                     count_tokens(tokenizer, input_ids)
    #                     for input_ids in batch["input_ids"]
    #                 ]
    #             )
    #             val_loss_recorder.add(epoch=epoch, step=step, loss=loss.item())
    #
    #             progress_bar.update(1)
    #             progress_bar.set_postfix(
    #                 {
    #                     "epoch": epoch + 1,
    #                     "lr/d*lr": optimizer.param_groups[0]["d"]
    #                     * optimizer.param_groups[0]["lr"],
    #                     "val_loss/average": val_loss_recorder.moving_average,
    #                     "loss/average": loss_recorder.moving_average,
    #                 }
    #             )
    #             # print(f"Epoch {epoch + 1}, Avg: {loss_recorder.moving_average:.4f}")
    #             accelerator.log(
    #                 {
    #                     "val_step": step,
    #                     "tokens_count": tokens_count,
    #                     "val_loss/average": val_loss_recorder.moving_average,
    #                     "lr/d*lr": optimizer.param_groups[0]["d"]
    #                     * optimizer.param_groups[0]["lr"],
    #                 },
    #                 step=global_step,
    #             )
    #
    # train_loader, val_loader = create_boolq_dataloaders(
    #     tokenizer,
    #     batch_size=args.batch_size,
    #     test_size=args.validation_split if args.validation_split is not None else 0,
    #     cache_dir=args.dataset_cache_dir,
    # )
    #
    # device_placement = [
    #     False,
    #     False,
    #     False if args.teacher_device is not None else True,
    #     False,
    #     False,
    #     False,
    # ]
    # train_loader, val_loader, teacher, student, adapter, optimizer = (
    #     accelerator.prepare(
    #         train_loader,
    #         val_loader,
    #         teacher,
    #         student,
    #         adapter,
    #         optimizer,
    #         device_placement=device_placement,
    #     )
    # )
    #
    # train(train_loader, val_loader, epochs=args.epochs)
    #
    # accelerator.end_training()
    #
    # end_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # metadata = compile_metadata(args)
    # metadata["end_timestamp"] = end_timestamp
    # metadata["tokens_count"] = str(get_tokens_count())
    # accelerator.print(f"Tokens count: {metadata['tokens_count']}")
    # save_file(
    #     adapter.state_dict(),
    #     f"adapter-t5-efficient-small-{end_timestamp}.safetensors",
    #     metadata=metadata,
    # )


def parse_dict(input_str):
    """Convert string input into a dictionary."""
    try:
        # Use ast.literal_eval to safely evaluate the string as a Python literal (dict)
        return ast.literal_eval(input_str)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid dictionary format: {input_str}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", default="google/t5-efficient-base")
    parser.add_argument("--student", default="google/t5-efficient-tiny")
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--teacher_device", default=None)
    parser.add_argument("--dataset_cache_dir", default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--save_every_n_epochs", type=int, default=None)
    parser.add_argument("--save_every_n_steps", type=int, default=None)
    parser.add_argument("--validation_split", type=float, default=0.1)
    parser.add_argument(
        "--optimizer_args",
        type=parse_dict,
        default={
            "weight_decay": 0.01,
            "eps": None,
            "use_orthograd": True,
            "use_adopt": True,
        },
    )
    parser.add_argument("--log_with", default="wandb")
    args = TrainingConfig(**vars(parser.parse_args()))
    main(args)
