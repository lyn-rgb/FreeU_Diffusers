"FreeU: Free Lunch in Diffusion U-Net" for Diffusers

The offical code https://github.com/ChenyangSi/FreeU

## Usage

### Image Pipelines

```python
import torch
from diffusers import StableDiffusionPipeline
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

### Video Pipelines

```python
import torch
from diffusers import TextToVideoSDPipeline
from diffusers.utils import export_to_video
from .free_lunch_utils import register_free_upblock3d, register_free_crossattn_upblock3d

model_id = "cerspense/zeroscope_v2_576w"
pipe = TextToVideoSDPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# -------- freeu block registration
register_free_upblock3d(pipe, b1=1.2, b2=1.4, s1=0.9, s2=0.2)
register_free_crossattn_upblock3d(pipe, b1=1.2, b2=1.4, s1=0.9, s2=0.2)
# -------- freeu block registration

prompt = "an astronaut riding a horse on mars"
video_frames = pipe(prompt, height=320, width=576, num_frames=30).frames
    
export_to_video(video_frames, "astronaut_rides_horse.mp4")
```

#### 28/09/23
Current version was successfully ran on diffusers v0.21.2.

#### 26/09/23
Note that it is supported and tested on diffusers v0.19.3. 
If you are using the latest diffusers, it is recommended to use the corresponding branch, but it has not been tested.