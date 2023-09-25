"FreeU: Free Lunch in Diffusion U-Net" for Diffusers

The offical code https://github.com/ChenyangSi/FreeU

## Usage
```python
from diffusers import StableDiffusionPipeline
import torch
from .free_lunch_utils import register_free_upblock2d, register_free_crossattn_upblock2d

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# -------- freeu block registration
register_free_upblock2d(pipe)
register_free_crossattn_upblock2d(pipe)
# -------- freeu block registration

prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt).images[0]  
    
image.save("astronaut_rides_horse.png")
```