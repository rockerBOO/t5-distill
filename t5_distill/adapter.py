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
from torch import nn


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        if m.bias:
            nn.init.zeros_(m.bias)


class ShapeAdapter(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.1):
        super().__init__()

        dims = self._get_progressive_dims(input_dim, output_dim)
        layers = []

        for i in range(len(dims) - 1):
            layers.extend(
                [
                    nn.Linear(dims[i], dims[i + 1], bias=False),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(dims[i + 1], dims[i + 1], bias=False),
                ]
            )

        print(layers)

        self.mlp = nn.Sequential(*layers)
        self.mlp.apply(init_weights)

    def _get_progressive_dims(self, input_dim, output_dim):
        dims = [input_dim]
        current = input_dim
        while current < output_dim:
            current *= 2
            dims.append(min(current, output_dim))
        return dims

    def forward(self, x):
        return self.mlp(x)
