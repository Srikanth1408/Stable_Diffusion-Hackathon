
#%%
from diffusers import DiffusionPipeline, StableDiffusionXLImg2ImgPipeline
import torch

prj_path = "/home/harshabommana/Hackathon"
model = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = DiffusionPipeline.from_pretrained(
    model,
    torch_dtype=torch.float16,
)
pipe.to("cuda")
pipe.load_lora_weights(prj_path, weight_name="pytorch_lora_weights.safetensors")
pipe.fuse_lora()

#%%
statements = [
    "Super ultra Realistic Indian roads with a bike heading left in rainy weather and heavy traffic during the early morning, creating human eye level image quality.",
    "Experience the authenticity of Indian roads as a car drives straight under the foggy sky with empty road conditions during the afternoon, offering unmatched visual quality.",
    "In the morning hours, a tractor moves right on realistic Indian roads, amidst sunny weather, providing a lifelike visual experience.",
    "Witness the precision of Indian roads with a truck navigating left through the messy road conditions under the cloudy sky during the evening, achieving remarkable image quality.",
    "Explore Indian roads at night with a bike racing right under the windy conditions, offering an ultra-realistic visual experience.",
    "A car travels straight on Indian roads in the afternoon, amidst sunny weather and clear road conditions, delivering an exceptional image quality.",
    "During the morning, a tractor plows through the foggy Indian road, heading straight with empty road conditions, presenting a vivid and detailed visual experience.",
    "On a foggy evening, a truck ventures straight down the road with heavy traffic, providing a human eye level image quality of Indian roads.",
    "Experience the charm of Indian roads as a bike zips through the messy road conditions during the morning, going left under the rainy sky, capturing super ultra-realistic image quality.",
    "In the early morning hours, a car travels right on Indian roads under the cloudy sky with empty road conditions, achieving an immersive and lifelike visual experience.",
    "Witness the realism of Indian roads with a tractor moving straight through the windy conditions during the noon, presenting unmatched image quality.",
    "During the night, a bike races straight on a rainy Indian road with heavy traffic, offering an ultra-realistic portrayal of human eye level image quality.",
    "Explore Indian roads like never before with a car driving right under the sunny sky during the evening, capturing exceptional image quality.",
    "A truck navigates through Indian roads in windy conditions, going left in the foggy weather during the afternoon, ensuring an immersive and detailed visual experience.",
    "In the evening, a tractor plows straight through realistic Indian roads, amidst messy road conditions and heavy traffic, offering remarkable image quality.",
    "On a cloudy morning, a bike speeds right on Indian roads with clear road conditions, providing an ultra-realistic visual experience.",
    "Experience the magic of Indian roads as a car drives straight under the sunny sky during the early morning, achieving human eye level image quality.",
    "During the afternoon, a truck moves left through the foggy Indian road, amidst messy road conditions, presenting unmatched image quality.",
    "Witness the precision of Indian roads with a tractor racing straight under the rainy sky with heavy traffic during the evening, delivering an immersive and detailed visual experience.",
    "Explore Indian roads at night with a bike zipping left under the windy conditions, capturing an ultra-realistic portrayal of human eye level image quality.",
    "A car navigates through Indian roads during the morning, heading straight under the cloudy sky with empty road conditions, ensuring an unmatched visual experience.",
    "In the early morning hours, a truck travels straight on Indian roads in sunny weather, offering a lifelike portrayal with remarkable image quality.",
    "Witness the realism of Indian roads as a bike races straight through the foggy weather during the noon, creating a vivid and detailed visual experience.",
    "On a rainy evening, a tractor moves right on Indian roads with heavy traffic, providing a super ultra-realistic image quality.",
    "Experience the charm of Indian roads as a car drives left under the messy road conditions during the early morning, amidst the windy weather, capturing human eye level image quality.",
    "During the afternoon, a bike speeds right on Indian roads with empty road conditions under the cloudy sky, delivering unmatched image quality.",
    "In the evening, a truck navigates straight through realistic Indian roads, creating a lifelike visual experience with remarkable image quality.",
    "On a foggy morning, a tractor plows through Indian roads, heading left under the rainy sky with heavy traffic, achieving super ultra-realistic image quality.",
    "Explore Indian roads like never before with a bike zipping straight under the sunny sky during the evening, offering an ultra-realistic visual experience.",
    "A car travels straight through Indian roads during the night, amidst clear road conditions and windy weather, capturing human eye level image quality.",
    "In the early morning hours, a bike races left on Indian roads under the cloudy sky, creating a vivid and detailed visual experience.",
    "Witness the precision of Indian roads with a truck moving straight through the messy road conditions during the afternoon, presenting unmatched image quality.",
    "On a rainy evening, a tractor ventures right on Indian roads in the foggy weather, providing a lifelike portrayal with remarkable image quality.",
    "Experience the authenticity of Indian roads as a car drives right under the windy conditions during the evening, achieving super ultra-realistic image quality.",
    "During the morning, a bike speeds left on Indian roads with heavy traffic, offering an ultra-realistic visual experience.",
    "In the afternoon, a truck navigates through Indian roads, heading straight under the rainy sky with empty road conditions, delivering unmatched image quality.",
    "On a foggy morning, a tractor plows straight through the messy road conditions on Indian roads, ensuring a vivid and detailed visual experience.",
    "Explore Indian roads at night with a bike racing straight under the cloudy sky with heavy traffic, capturing human eye level image quality.",
    "A car travels left on Indian roads in the early morning under the windy conditions, providing a super ultra-realistic portrayal with remarkable image quality.",
    "Witness the magic of Indian roads as a truck moves straight through the sunny weather during the morning, achieving an immersive and lifelike visual experience.",
    "In the evening, a bike races straight on Indian roads, amidst the rainy sky with clear road conditions, presenting a remarkable image quality.",
    "Experience the realism of Indian roads as a tractor navigates straight through the foggy Indian road with heavy traffic during the afternoon, offering unmatched image quality.",
    "During the night, a car drives straight on Indian roads under the cloudy sky with messy road conditions, ensuring an ultra-realistic visual experience.",
    "On a foggy evening, a bike zips right on Indian roads, capturing the windy conditions, delivering an immersive and detailed portrayal with human eye level image quality.",
    "Witness the precision of Indian roads as a truck races left under the sunny sky during the evening, offering remarkable image quality.",
    "In the early morning hours, a tractor plows straight through Indian roads, amidst the rainy weather with empty road conditions, ensuring a lifelike visual experience.",
    "Explore Indian roads like never before with a bike navigating left on a messy road during the night, capturing super ultra-realistic image quality.",
    "A car travels straight on Indian roads under the cloudy sky with heavy traffic during the afternoon, delivering unmatched image quality.",
    "In the morning, a truck moves right on Indian roads, amidst the foggy weather, creating an ultra-realistic visual experience.",
    "On a rainy evening, a bike speeds straight through the realistic Indian road with empty road conditions, offering remarkable image quality.",
    "Experience the charm of Indian roads as a tractor ventures left under the windy conditions during the evening, capturing human eye level image quality."
]

statements1 = [
    "Super ultra Realistic Indian roads come to life as a bike gracefully meanders to the right on a sunny morning, amidst bustling traffic, creating imagery so vivid it's like being there.",
    "Experience the picturesque charm of Indian congested roads as a car drives straight beneath the cloudy sky, through a lively market during the afternoon, capturing exceptional human-eye-level visuals.",
    "In the early morning, a tractor navigates right on realistic Indian roads, amidst sunny weather, providing a serene and lifelike visual experience.",
    "Witness the precision of Indian roads with a truck moving left through the market's chaotic road conditions under the clear blue sky during the evening, achieving remarkable image quality.",
    "Explore Indian busy roads at night with a bike racing straight, passing through vibrant street markets under the starry sky, offering an ultra-realistic and immersive visual experience.",
    "A car cruises through Indian roads in the afternoon, amidst sunny weather, and the open market stalls on the side showcase vivid colors and life-like details, delivering an exceptional image quality.",
    "During the morning rush, a tractor plows through the foggy Indian road, heading straight past bustling market stalls, presenting a vivid and detailed visual experience.",
    "On a foggy evening, a truck ventures straight through the busy market with heavy traffic, providing a human-eye-level image quality of Indian roads packed with vibrant market activity.",
    "Feel the charm of Indian roads as a bike maneuvers left through the lively market during the rainy morning, the colorful market umbrellas contrasting with the gray sky, capturing super ultra-realistic image quality.",
    "In the early morning hours, a car travels right on Indian roads under the cloudy sky, navigating through a bustling market, achieving an immersive and lifelike visual experience.",
    "Witness the realism of Indian roads with a tractor moving straight through the bustling market in windy conditions during the noon, presenting unmatched image quality.",
    "Explore the market life of Indian roads during the night, as a bike races straight through the rainy Indian road, amidst heavy traffic and market stalls, offering an ultra-realistic portrayal of human eye level image quality.",
    "Drive into the vibrant evening with a car going right on Indian roads, passing through a lively market, capturing exceptional image quality with vibrant market scenes.",
    "A truck navigates through Indian roads in windy conditions, heading left through the market's vibrant scene during the foggy afternoon, ensuring an immersive and detailed visual experience.",
    "In the evening, a tractor plows through realistic Indian roads, navigating straight through the market, amidst messy road conditions and heavy traffic, offering remarkable image quality with vibrant market scenes.",
    "On a cloudy morning, a bike speeds right on Indian roads with clear road conditions, passing through a bustling market, providing an ultra-realistic visual experience with vibrant market scenes.",
    "Experience the market's hustle and bustle on Indian roads as a car drives straight under the sunny sky during the early morning, passing by vibrant market stalls, showcasing human eye level image quality.",
    "Journey through a sunny afternoon as a truck moves left along a bustling Indian market, amidst foggy weather, highlighting vibrant market scenes, and capturing exquisite image quality.",
    "Witness the precision of Indian roads with a tractor racing straight under the rainy sky, passing through a lively market, creating an immersive and detailed visual experience with vibrant market scenes.",
    "Embark on a captivating night ride with a bike darting straight through a rainy Indian road, surrounded by bustling market activity and heavy traffic, offering an ultra-realistic portrayal with human eye level image quality and vibrant market scenes.",
    "Explore the vibrant market life on Indian roads at night as a car travels left under the messy road conditions, embracing the windy weather and the bustling market, creating an immersive visual experience with vibrant market scenes.",
    "In the afternoon, a bike accelerates right on Indian roads with clear road conditions under the cloudy sky, capturing an unmatched image quality with vibrant market scenes.",
    "A tractor ventures into the heart of Indian roads, moving left through the sunny weather during the morning, passing through a bustling market with vibrant market scenes, capturing remarkable image quality.",
    "During the early morning, a truck cruises straight through the foggy Indian road, emphasizing the empty road conditions and the vibrant market life, providing a vivid and detailed visual experience.",
    "Witness the realism of Indian roads as a bike races right under windy conditions during the evening, passing through a bustling market with vibrant market scenes, delivering exceptional image quality.",
    "Experience the vibrant market life on Indian roads as a car glides straight under the foggy afternoon sky, showcasing human eye level image quality amidst bustling market activity.",
    "In the evening, a tractor navigates straight through the messy road conditions on realistic Indian roads, surrounded by heavy traffic and vibrant market stalls, ensuring remarkable image quality with vibrant market scenes.",
    "On a foggy morning, a truck travels right on Indian roads with a foggy sky overhead, emphasizing heavy traffic and the vibrant market life, achieving super ultra-realistic image quality with vibrant market scenes.",
    "Explore Indian roads like never before with a bike zipping straight under the sunny noon sky, passing through a bustling market, offering unmatched image quality with vibrant market scenes.",
    "A car travels straight through Indian roads during the night, accompanied by clear road conditions and windy weather, passing through a bustling market with vibrant market scenes, presenting a human eye level image quality.",
    "In the early morning hours, a bike races left on Indian roads under the cloudy sky, creating a vivid and detailed visual experience while passing through bustling market stalls with vibrant market scenes.",
    "Witness the precision of Indian roads with a truck moving straight through the messy road conditions during the afternoon, highlighting unmatched image quality with vibrant market scenes.",
    "On a rainy evening, a tractor ventures right on Indian roads amidst the foggy weather, passing through a bustling market with vibrant market scenes, providing a lifelike portrayal with remarkable image quality.",
    "Experience the authenticity of Indian roads as a car drives right under the windy conditions during the evening, passing through a bustling market with vibrant market scenes, achieving super ultra-realistic image quality.",
    "During the morning, a bike speeds left on Indian roads with heavy traffic, passing through a bustling market, offering an ultra-realistic visual experience with vibrant market scenes."
]
#j=100
#for i in statements1:
prompt = "ultra realistic indian road with vehicles waiting to cross railway line, 4k image"
#j=j+1
seed = 4575
generator = torch.Generator("cuda").manual_seed(seed)
image = pipe(prompt=prompt, generator=generator).images[0]
image.save(f"/home/harshabommana/Hackathon/generated_images/generated_image500.png")
# %%


