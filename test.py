import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Dataset_own, Trainer


if torch.cuda.is_available():
    # Print the number of CUDA devices available
    num_gpus = torch.cuda.device_count()
    print(f"Number of CUDA devices available: {num_gpus}")
    
    # Iterate over available CUDA devices and print their names
    for i in range(num_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        print(f"GPU {i}: {gpu_name}")
else:
    print("CUDA is not available.")
model = Unet(
    dim = 128,
    dim_mults = (1, 2, 4, 8),
    flash_attn = True,
    attn_dim_head = 32,
    attn_heads = 4,
)

diffusion = GaussianDiffusion(
    model,
    image_size = 128,
    timesteps = 1000    # number of steps
)

trainer = Trainer(diffusion_model=diffusion,
                    folder = "data",
                    gradient_accumulate_every=4,
                    train_batch_size=16,
                    train_lr=1e-5,
                    train_num_steps=7_500,
                    save_and_sample_every=1_000,
                    calculate_fid=False)


trainer.train()
trainer.save("test_15_000")

