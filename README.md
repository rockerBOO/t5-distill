# T5 Distillation

Train T5 distillation models. Current goal to provide training tool for T5 Encoder distillation for use in T5 encoder enhanced models like Flux, Stable Diffusion 3.

- T5 Encoder distillation via Adapter projection model

The idea is we can make a projection from the student model encoder logits to the teacher model encoder logits. This will allow us to utilize pre-trained T5 models of smaller sizes and project those logits/embeddings to the larger teacher model. This allows us to bridge the gap into drop in replacement for larger models. For usage in models like Flux, Stable Diffusion 3.

## Install

Install [uv](https://docs.astral.sh/uv/) for package management.

Using uv, sync the dependencies.

```
uv sync --frozen --no-install-project --no-dev
```

_NOTE_ if you are having any issues, let me know. I am trying out `uv` but interested in any issues with syncing dependencies or usage.

# Usage

**Note** A work in progress so you should also read the code and be aware of how it works. Generally settings should be applied as expected but hasn't gone through enough testing to be sure. PR's are open for improvements or adding better functionality.

```sh
# Teacher model we will make an adapter from
teacher="google/t5-efficient-base"
# Student model we will make an adapter for
student="google/t5-efficient-tiny"
seed=1234
batch_size=32
epochs=15
# Optimizer arguments for ProdigyPlusScheduleFree optimizer
optimizer_args="{'weight_decay': 0.01,'eps':None,'use_orthograd':True,'use_adopt':True}"
log_with="wandb"

accelerate launch main.py --teacher $teacher --student $student --seed $seed --batch_size $batch_size --epochs $epochs --optimizer_args "$optimizer_args"
```

# Development

- Pyright typechecking (lossly)
- Ruff formatting
- uv for dependency management

# Credits

In no way directly associated with these companies and not in any partnership.

- Thanks to HuggingFace for providing models weights and model code.
- Thanks to Google for providing T5 models.
- Thanks to Kohya and contributors of sd-scripts for some functionality in this code
- Thanks to blackforestlabs and stabilityai for creating Flux and Stable Diffusion 3
