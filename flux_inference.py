import torch
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from typing import Optional
from safetensors.torch import load_model

from transformers import T5EncoderModel
from torch import nn


class ShapeAdapter(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Dropout(dropout),
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
            nn.Dropout(dropout),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x):
        return self.mlp(x)


class TextEncoder(T5EncoderModel):
    def __init__(self, config):
        super().__init__(config)

    def set_shape_adapter(self, shape_adapter: ShapeAdapter):
        self.shape_adapter = shape_adapter

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        x = super().forward(
            input_ids,
            attention_mask,
            head_mask,
            inputs_embeds,
            output_attentions,
            output_hidden_states,
            return_dict,
        )
        last_hidden_state = self.shape_adapter(x.last_hidden_state)
        x.last_hidden_state = last_hidden_state
        return x


text_encoder = TextEncoder.from_pretrained(
    "google/t5-efficient-tiny",
    use_safetensors=True,
)
shape_adapter = ShapeAdapter(256, 4096)
load_model(shape_adapter, "adapter-t5-efficient-small-2025-01-15_01-41-51.safetensors")
text_encoder.set_shape_adapter(shape_adapter)

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    text_encoder_2=text_encoder,
    torch_dtype=torch.float16,
)
pipe.enable_sequential_cpu_offload()
# pipe.enable_model_cpu_offload()
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()

pipe.to(torch.float16)

# prompt = "A cat holding a sign that says hello world"
prompt = "a beautiful amazing woman"
image = pipe(
    prompt,
    height=768,
    width=1024,
    guidance_scale=3.5,
    num_inference_steps=15,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(0),
).images[0]
image.save("flux-dev.png")
