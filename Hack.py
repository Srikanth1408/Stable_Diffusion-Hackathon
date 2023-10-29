#%%

import os
!pip install -U autotrain-advanced > install_logs.txt
!autotrain setup --colab > setup_logs.txt



# %%
#@markdown ---
#@markdown #### Project Config
project_name = 'my_dreambooth_project' # @param {type:"string"}
model_name = 'stabilityai/stable-diffusion-xl-base-1.0' # @param ["stabilityai/stable-diffusion-xl-base-1.0", "runwayml/stable-diffusion-v1-5", "stabilityai/stable-diffusion-2-1", "stabilityai/stable-diffusion-2-1-base"]
prompt = 'Indian roads' # @param {type: "string"}

#@markdown ---
#@markdown #### Push to Hub?
#@markdown Use these only if you want to push your trained model to a private repo in your Hugging Face Account
#@markdown If you dont use these, the model will be saved in Google Colab and you are required to download it manually.
#@markdown Please enter your Hugging Face write token. The trained model will be saved to your Hugging Face account.
#@markdown You can find your token here: https://huggingface.co/settings/tokens
push_to_hub = False # @param ["False", "True"] {type:"raw"}
hf_token = "hf_XXX" #@param {type:"string"}
repo_id = "username/repo_name" #@param {type:"string"}

#@markdown ---
#@markdown #### Hyperparameters
learning_rate = 1e-4 # @param {type:"number"}
num_steps = 500 #@param {type:"number"}
batch_size = 1 # @param {type:"slider", min:1, max:32, step:1}
gradient_accumulation = 4 # @param {type:"slider", min:1, max:32, step:1}
resolution = 1024 # @param {type:"slider", min:128, max:1024, step:128}
use_8bit_adam = True # @param ["False", "True"] {type:"raw"}
use_xformers = True # @param ["False", "True"] {type:"raw"}
use_fp16 = True # @param ["False", "True"] {type:"raw"}
train_text_encoder = False # @param ["False", "True"] {type:"raw"}
gradient_checkpointing = True # @param ["False", "True"] {type:"raw"}

os.environ["PROJECT_NAME"] = project_name
os.environ["MODEL_NAME"] = model_name
os.environ["PROMPT"] = prompt
os.environ["PUSH_TO_HUB"] = str(push_to_hub)
os.environ["HF_TOKEN"] = hf_token
os.environ["REPO_ID"] = repo_id
os.environ["LEARNING_RATE"] = str(learning_rate)
os.environ["NUM_STEPS"] = str(num_steps)
os.environ["BATCH_SIZE"] = str(batch_size)
os.environ["GRADIENT_ACCUMULATION"] = str(gradient_accumulation)
os.environ["RESOLUTION"] = str(resolution)
os.environ["USE_8BIT_ADAM"] = str(use_8bit_adam)
os.environ["USE_XFORMERS"] = str(use_xformers)
os.environ["USE_FP16"] = str(use_fp16)
os.environ["TRAIN_TEXT_ENCODER"] = str(train_text_encoder)
os.environ["GRADIENT_CHECKPOINTING"] = str(gradient_checkpointing)



#%%


import torch
print(torch.__version__)



# %%
# Inference
# this is the inference code that you can use after you have trained your model
# Unhide code below and change prj_path to your repo or local path (e.g. my_dreambooth_project)
#
#
#
from diffusers import DiffusionPipeline, StableDiffusionXLImg2ImgPipeline
import torch

prj_path = "/Users/karthikeyan-mohanraj/Downloads"
model = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = DiffusionPipeline.from_pretrained(
    model,
    torch_dtype=torch.float16,
)
pipe.to("mps")
pipe.load_lora_weights(prj_path, weight_name="pytorch_lora_weights.safetensors")

#refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
#    "stabilityai/stable-diffusion-xl-refiner-1.0",
#    torch_dtype=torch.float16,
#)
#refiner.to("mps")


#%%



# %%
prompt = "realistic indian road image, 4k"

seed = 42
generator = torch.Generator("mps").manual_seed(seed)
image = pipe(prompt=prompt, generator=generator).images[0]
#image = pipe(prompt=prompt).images[0]
#image = refiner(prompt=prompt, generator=generator, image=image).images[0]
image.save(f"generated_image.png")
# %%
# %%
prompt = "realistic indian road image with vehicles and traffic light, 4k image"

seed = 42
generator = torch.Generator("mps").manual_seed(seed)
image = pipe(prompt=prompt, generator=generator).images[0]
#image = pipe(prompt=prompt).images[0]
#image = refiner(prompt=prompt, generator=generator, image=image).images[0]
image.save(f"generated_image1.png")



image.show()

# %%
prompt = "realistic Indian village road with bike,tractors surrounded by crops and animals in rainy season, 2k image  "

seed = 42
generator = torch.Generator("mps").manual_seed(seed)
image = pipe(prompt=prompt, generator=generator).images[0]
#image = pipe(prompt=prompt).images[0]
#image = refiner(prompt=prompt, generator=generator, image=image).images[0]
image.save(f"generated_image2.png")



image.show()
# %%
prompt = "realistic Indian village road with bike,tractors surrounded by crops and animals in rainy season, 2k image  "

seed = 42
generator = torch.Generator("mps").manual_seed(seed)
image = pipe(prompt=prompt, generator=generator).images[0]
#image = pipe(prompt=prompt).images[0]
#image = refiner(prompt=prompt, generator=generator, image=image).images[0]
image.save(f"generated_image2.png")

# %%
prompt = "realistic Indian city road car taking right turn and bike taking left turn, 2k image  "

seed = 42
generator = torch.Generator("mps").manual_seed(seed)
image = pipe(prompt=prompt, generator=generator).images[0]
#image = pipe(prompt=prompt).images[0]
#image = refiner(prompt=prompt, generator=generator, image=image).images[0]
image.save(f"generated_image3.png")
# %%


prompt = "realistic indian crowded roads with vehicles and people,  4k image  "

seed = 42
generator = torch.Generator("mps").manual_seed(seed)
image = pipe(prompt=prompt, generator=generator).images[0]
#image = pipe(prompt=prompt).images[0]
#image = refiner(prompt=prompt, generator=generator, image=image).images[0]
image.save(f"generated_image4.png")
# %%
image.show()
# %%
prompt = "ultra realistic indian night road with vehicles waiting to cross railway line,  4k image  "

seed = 42
generator = torch.Generator("mps").manual_seed(seed)
image = pipe(prompt=prompt, generator=generator).images[0]
#image = pipe(prompt=prompt).images[0]
#image = refiner(prompt=prompt, generator=generator, image=image).images[0]
image.save(f"generated_image5.png")

image.show()

# %%
prompt = "ultra realistic indian scenario where car is overriding on a water filled pot hole,  4k image  "

seed = 42
generator = torch.Generator("mps").manual_seed(seed)
image = pipe(prompt=prompt, generator=generator).images[0]
#image = pipe(prompt=prompt).images[0]
#image = refiner(prompt=prompt, generator=generator, image=image).images[0]
image.save(f"generated_image6.png")

image.show()

# %%
prompt = "ultra realistic indian road with vehicles waiting to cross railway line,  4k image  "

seed = 42
generator = torch.Generator("mps").manual_seed(seed)
image = pipe(prompt=prompt, generator=generator).images[0]
#image = pipe(prompt=prompt).images[0]
#image = refiner(prompt=prompt, generator=generator, image=image).images[0]
image.save(f"generated_image10.png")

image.show()


# %%
