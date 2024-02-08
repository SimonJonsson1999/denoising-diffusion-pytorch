import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Dataset_own, Trainer

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    flash_attn = True
)

diffusion = GaussianDiffusion(
    model,
    image_size = 128,
    timesteps = 1000    # number of steps
)

# training_images = torch.rand(8, 3, 128, 128) # images are normalized from 0 to 1
# loss = diffusion(training_images)

# loss.backward()

# # after a lot of training

# sampled_images = diffusion.sample(batch_size = 4)
# sampled_images.shape # (4, 3, 128, 128)


# dataset = Dataset_own(folder="data", image_size=256)
trainer = Trainer(diffusion_model=diffusion, folder = "data")
trainer.train()
trainer.save("test")