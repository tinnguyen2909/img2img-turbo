import os
import torch
import numpy as np
from pathlib import Path
import argparse
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm

from cut_turbo import (
    VAE_encode, 
    VAE_decode, 
    initialize_unet, 
    initialize_vae, 
    PatchNCELoss, 
    PatchSampleF, 
    CUT_Turbo
)


def create_test_image(size=(3, 64, 64)):
    """Create a test image with a basic pattern."""
    # Create a simple image with a gradient pattern
    channels, height, width = size
    img = torch.zeros(size)
    for i in range(height):
        for j in range(width):
            # Create a simple pattern (gradient + circle)
            dist_from_center = ((i - height // 2) ** 2 + (j - width // 2) ** 2) ** 0.5
            img[0, i, j] = i / height  # Red channel: vertical gradient
            img[1, i, j] = j / width   # Green channel: horizontal gradient
            img[2, i, j] = 1 - dist_from_center / (height // 2)  # Blue channel: radial gradient
    
    # Normalize to [-1, 1] range
    img = img * 2 - 1
    return img


def test_vae_components():
    """Test VAE encode and decode operations."""
    print("\n=== Testing VAE Components ===")
    
    # Initialize VAE
    print("Initializing VAE...")
    vae, _ = initialize_vae(lora_rank=4, return_lora_module_names=True)
    
    # Create VAE encode/decode modules
    vae_enc = VAE_encode(vae)
    vae_dec = VAE_decode(vae)
    
    # Create test image
    img = create_test_image((3, 256, 256)).unsqueeze(0).cuda()  # Add batch dimension
    print(f"Input image shape: {img.shape}")
    
    # Encode image
    print("Encoding image...")
    latent = vae_enc(img)
    print(f"Latent shape: {latent.shape}")
    
    # Decode latent
    print("Decoding latent...")
    reconstructed = vae_dec(latent)
    print(f"Reconstructed image shape: {reconstructed.shape}")
    
    # Calculate reconstruction error
    mse = torch.mean((img - reconstructed) ** 2).item()
    print(f"Reconstruction MSE: {mse:.6f}")
    
    # Save input and reconstructed images for visual inspection
    save_image(img[0], "vae_input.png")
    save_image(reconstructed[0], "vae_output.png")
    
    print("VAE components test completed. Check vae_input.png and vae_output.png for visual results.")
    return mse < 0.1  # Expect low reconstruction error


def test_patchnce_loss():
    """Test PatchNCE loss computation."""
    print("\n=== Testing PatchNCE Loss ===")
    
    # Define NCE layers and create loss function
    nce_layers = [0, 1, 2]
    criterion = PatchNCELoss(nce_layers=nce_layers, nce_temp=0.07)
    
    # Create identical and different feature maps
    batch_size = 2
    nc = 64
    feat_identical = [torch.rand(batch_size, nc, 8, 8).cuda() for _ in range(len(nce_layers))]
    feat_different = [torch.rand(batch_size, nc, 8, 8).cuda() for _ in range(len(nce_layers))]
    
    # Compute losses
    print("Computing PatchNCE loss for identical features...")
    loss_identical = criterion(feat_identical, feat_identical)
    
    print("Computing PatchNCE loss for different features...")
    loss_different = criterion(feat_identical, feat_different)
    
    print(f"Loss for identical features: {loss_identical.item():.6f}")
    print(f"Loss for different features: {loss_different.item():.6f}")
    
    # Losses should follow: loss_identical < loss_different
    print(f"PatchNCE loss test {'PASSED' if loss_identical.item() < loss_different.item() else 'FAILED'}")
    return loss_identical.item() < loss_different.item()


def test_patch_sample():
    """Test PatchSampleF module for feature extraction."""
    print("\n=== Testing PatchSampleF Module ===")
    
    # Initialize UNet
    print("Initializing UNet...")
    unet, _, _, _ = initialize_unet(lora_rank=8, return_lora_module_names=True)
    
    # Define NCE layers and create PatchSampleF
    nce_layers = [0, 4, 8]
    patch_sample = PatchSampleF(nce_layers=nce_layers, use_mlp=True, nc=256).cuda()
    
    # Create input tensors
    z = torch.randn(2, 4, 32, 32).cuda()
    text_emb = torch.randn(2, 77, 768).cuda()
    timesteps = torch.tensor([999, 999]).cuda()
    
    # Extract features
    print("Extracting features using PatchSampleF...")
    features = patch_sample(unet, z, text_emb, timesteps)
    
    # Check features
    print(f"Number of feature maps: {len(features)}")
    for i, feat in enumerate(features):
        print(f"Feature {i} shape: {feat.shape}")
    
    # Check dimensions and properties
    all_valid = all(feat.shape[1] == 256 for feat in features)  # All should have nc=256 channels
    print(f"PatchSampleF test {'PASSED' if all_valid else 'FAILED'}")
    return all_valid


def test_cut_turbο_e2e():
    """Test the full CUT_Turbo model end-to-end."""
    print("\n=== Testing CUT_Turbo End-to-End ===")
    
    # Initialize the model
    print("Initializing CUT_Turbo model...")
    model = CUT_Turbo(lora_rank_unet=8, lora_rank_vae=4)
    model.timesteps = torch.tensor([999], device="cuda").long()
    
    # Create a test image
    img = create_test_image((3, 256, 256)).unsqueeze(0).cuda()
    print(f"Input image shape: {img.shape}")
    
    # Define a caption
    caption = "A beautiful landscape with mountains and trees"
    
    # Generate an image using the model
    print("Generating image with the model...")
    with torch.no_grad():
        output = model(img, caption=caption)
    
    print(f"Output image shape: {output.shape}")
    
    # Save the input and output images
    save_image(img[0], "cut_input.png")
    save_image(output[0], "cut_output.png")
    
    # Also test the specific NCE loss computation
    print("Testing PatchNCE loss computation...")
    patch_sample = model.patch_sample
    criterionNCE = model.criterionNCE
    
    # Get latent for source image
    with torch.no_grad():
        enc_src = model.vae_enc(img)
    
    # Compute PatchNCE loss
    nce_loss = CUT_Turbo.compute_nce_loss(
        img, output, enc_src, model.unet, patch_sample, criterionNCE, 
        model.timesteps.repeat(img.shape[0]), 
        model.text_encoder(model.tokenizer(
            caption, max_length=model.tokenizer.model_max_length,
            padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids.cuda())[0]
    )
    
    print(f"PatchNCE loss: {nce_loss.item():.6f}")
    
    print("CUT_Turbo end-to-end test completed. Check cut_input.png and cut_output.png for visual results.")
    return output.shape == img.shape and nce_loss.item() > 0


def save_image(tensor, filename):
    """Save a tensor as an image file."""
    # Convert to PIL image and save
    pil_img = transforms.ToPILImage()(tensor.cpu().float() * 0.5 + 0.5)
    pil_img.save(filename)
    print(f"Saved image to {filename}")


def main():
    parser = argparse.ArgumentParser(description="Run end-to-end tests for CUT_Turbo model")
    parser.add_argument('--skip-vae', action='store_true', help='Skip VAE component tests')
    parser.add_argument('--skip-patchnce', action='store_true', help='Skip PatchNCE loss tests')
    parser.add_argument('--skip-patchsample', action='store_true', help='Skip PatchSampleF tests')
    parser.add_argument('--skip-e2e', action='store_true', help='Skip end-to-end model tests')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run tests
    results = {}
    
    if not args.skip_vae:
        results['vae'] = test_vae_components()
    
    if not args.skip_patchnce:
        results['patchnce'] = test_patchnce_loss()
    
    if not args.skip_patchsample:
        results['patchsample'] = test_patch_sample()
    
    if not args.skip_e2e:
        results['e2e'] = test_cut_turbο_e2e()
    
    # Print summary
    print("\n=== Test Summary ===")
    for test_name, passed in results.items():
        print(f"{test_name}: {'PASSED' if passed else 'FAILED'}")
    
    all_passed = all(results.values())
    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    main() 