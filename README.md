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
register_free_upblock2d(pipe, b1=1.2, b2=1.4, s1=0.9, s2=0.2)
register_free_crossattn_upblock2d(pipe, b1=1.2, b2=1.4, s1=0.9, s2=0.2)
# -------- freeu block registration

prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt).images[0]  
    
image.save("astronaut_rides_horse.png")
```

Note that it is supported and tested on diffusers v0.19.3. 
If you are using the latest diffusers, it is recommended to use the corresponding branch, but it has not been tested.
