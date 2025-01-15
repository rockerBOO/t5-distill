from typing import Dict, Optional, Union
from dataclasses import dataclass, asdict


@dataclass
class TrainingConfig:
    optimizer_args: Dict
    teacher: str
    student: str
    batch_size: int
    epochs: int
    validation_split: Optional[float]
    teacher_device: Optional[str]
    dataset_cache_dir: Optional[str]
    steps: Optional[int]
    save_every_n_epochs: Optional[int]
    save_every_n_steps: Optional[int]
    seed: Optional[int]
    log_with: Optional[Union[str, list[str]]]


def compile_metadata(args) -> dict[str, str]:
    metadata = asdict(args)
    for k in metadata.keys():
        metadata[k] = str(metadata[k])

    return metadata
