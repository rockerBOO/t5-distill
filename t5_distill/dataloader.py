from datasets import load_dataset, DatasetDict
from torch.utils.data import DataLoader

from transformers import (
    DataCollatorWithPadding,
    T5Tokenizer,
)


def create_c4_dataloaders(tokenizer, generator, batch_size, cache_dir=None):
    dataset = load_dataset(
        "allenai/c4",
        revision="convert/parquet",
        data_dir="en",
        cache_dir=cache_dir,
    )

    def preprocess(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding=True,
            return_tensors="pt",
        )

    assert isinstance(dataset, DatasetDict)

    tokenized_dataset = dataset.map(
        preprocess, batched=True, remove_columns=dataset["train"].column_names
    )

    data_collator = DataCollatorWithPadding(tokenizer)

    train_loader = DataLoader(
        tokenized_dataset["train"],
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=data_collator,
    )

    val_loader = DataLoader(
        tokenized_dataset["validation"],
        batch_size=batch_size,
        collate_fn=data_collator,
    )

    return train_loader, val_loader


def create_boolq_dataloaders(
    tokenizer: T5Tokenizer,
    batch_size=32,
    test_size=0.1,
    generator=None,
    cache_dir=None,
):
    dataset = load_dataset("google/boolq", cache_dir=cache_dir)

    assert isinstance(dataset, DatasetDict)

    # Filter for only true answers
    dataset = dataset.filter(lambda x: x["answer"] is True)

    def preprocess(examples):
        questions = examples["question"]

        # Encode questions as input
        model_inputs = tokenizer(
            questions,
            truncation=True,
            max_length=512,
            padding=True,
            return_tensors="pt",
        )

        return model_inputs

    tokenized_train = dataset["train"].map(
        preprocess, batched=True, remove_columns=dataset["train"].column_names
    )

    tokenized_val = dataset["validation"].map(
        preprocess, batched=True, remove_columns=dataset["train"].column_names
    )

    data_collator = DataCollatorWithPadding(tokenizer)

    train_loader = DataLoader(
        tokenized_train,
        shuffle=True,
        generator=generator,
        batch_size=batch_size,
        collate_fn=data_collator,
    )

    val_loader = DataLoader(
        tokenized_val,
        batch_size=batch_size,
        collate_fn=data_collator,
    )

    return train_loader, val_loader
