# Copyright 2025 Dave Lage (rockerBOO)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
