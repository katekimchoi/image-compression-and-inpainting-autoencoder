# utils.py
import matplotlib.pyplot as plt   
import torch 
import torchvision.utils as vutils
import numpy as np
import random
import os

# Show a sample of the dataset #
def imshow(imgs):
    imgs = imgs / 2 + 0.5  
    npimgs = imgs.numpy()
    plt.imshow(np.transpose(npimgs, (1, 2, 0)))
    plt.show()

# Show compression model reconstructions #
def show_compression_reconstructions(model, loader, device, n_show=8, fname=None):
    model.eval()
    with torch.no_grad():
        imgs, _ = next(iter(loader))
        imgs = imgs.to(device)
        recons = model(imgs)

        # Reshape (if decoder outputs (B, 3072))
        if recons.ndim == 2:
            recons = recons.view(-1, 3, 32, 32)

        imgs = imgs.cpu()
        recons = recons.cpu()

        # Unnormalize manually so originals and recons are comparable
        imgs   = imgs * 0.5 + 0.5
        recons = recons * 0.5 + 0.5

        comp = torch.cat([imgs[:n_show], recons[:n_show]], dim=0)
        grid = vutils.make_grid(comp, nrow=n_show)

        plt.figure(figsize=(n_show*1.2, 2.4))
        plt.axis("off")
        plt.title("Top: originals | Bottom: reconstructions")
        plt.imshow(np.transpose(grid.numpy(), (1, 2, 0)))
        if fname:
            plt.savefig(fname, bbox_inches="tight")
        plt.close()
    model.train()

# Show inpainting model reconstructions #
def show_inpainting_reconstructions(model, loader, device, n_show=8, fname=None):
    model.eval()
    with torch.no_grad():
        imgs, _ = next(iter(loader))
        imgs = imgs.to(device)
        patch_imgs = remove_patch(imgs, device)
        recons = model(patch_imgs)

        # Reshape
        if recons.ndim == 2:
            recons = recons.view(-1, 3, 32, 32)

        imgs = imgs.cpu()
        patch_imgs = patch_imgs.cpu()
        recons = recons.cpu()

        # Unnormalize manually so originals, patch images, and recons are comparable
        def unnormalize(tensor):
            return tensor * 0.5 + 0.5  # Since normalized using (0.5, 0.5, 0.5)
        imgs = unnormalize(imgs[:n_show])
        patch_imgs = unnormalize(patch_imgs[:n_show])
        recons = unnormalize(recons[:n_show])

        comp = torch.cat([imgs[:n_show], patch_imgs[:n_show], recons[:n_show]], dim=0)
        grid = vutils.make_grid(comp, nrow=n_show)  

        plt.figure(figsize=(n_show*1.2, 2.4))
        plt.axis("off")
        plt.title("Top: originals | Middle: patch images | Bottom: reconstructions")
        plt.imshow(np.transpose(grid.numpy(), (1, 2, 0)))
        if fname:
            plt.savefig(fname, bbox_inches="tight")
        plt.close()
    model.train()

# Remove a patch for image inpainting #
# Current patch size constraints: 20% of height and 20% of width maximum
def remove_patch(images, device):
    patch_height_max = int(0.2 * images.shape[2])
    patch_width_max = int(0.2 * images.shape[3])
    patch_height = random.randint(1, patch_height_max)
    patch_width = random.randint(1, patch_width_max)
    
    patch_height_end = random.randint(patch_height, images.shape[2])
    patch_width_end = random.randint(patch_width, images.shape[3])
    patch_height_start = patch_height_end - patch_height
    patch_width_start = patch_width_end - patch_width

    patch_images = images.clone().to(device)
    patch_images[:, :, patch_height_start:patch_height_end, patch_width_start:patch_width_end] = 0
    
    return patch_images

# Save and load model #
def save_checkpoint(model, optimizer, epoch, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }, path)

def load_checkpoint(model, optimizer, path, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt.get("epoch", None)
