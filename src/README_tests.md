# Testing CUT-Turbo

This document provides information about the test suite for the CUT-Turbo model (Contrastive Unpaired Translation with Stable Diffusion Turbo).

## Test Files

1. **test_cut_turbo.py**: Unit tests for individual components using mock objects
2. **test_cut_turbo_e2e.py**: End-to-end tests using small real models

## What's Being Tested

### Unit Tests (`test_cut_turbo.py`)

These tests use the `unittest` framework with mock objects to test each component in isolation:

- **VAE_encode**: Tests the encoding part of the VAE
- **VAE_decode**: Tests the decoding part of the VAE
- **initialize_unet**: Tests UNet initialization with LoRA adapters
- **initialize_vae**: Tests VAE initialization with skip connections and LoRA
- **PatchNCELoss**: Tests the contrastive loss function with various inputs
- **PatchSampleF**: Tests the patch sampling module for feature extraction
- **CUT_Turbo**: Tests the main model class including:
  - Initialization
  - Forward pass
  - PatchNCE loss computation
  - Parameter collection
  - Checkpoint loading

### End-to-End Tests (`test_cut_turbo_e2e.py`)

These tests use actual (smaller) models to test the full pipeline:

- **VAE Components**: Tests encoding and decoding with a real VAE
- **PatchNCE Loss**: Tests the loss computation with real feature maps
- **PatchSampleF**: Tests feature extraction from a real UNet
- **CUT-Turbo End-to-End**: Tests the full model, translating an image and computing the loss

## Running the Tests

### Unit Tests

Run the unit tests with:

```bash
python src/test_cut_turbo.py
```

These tests use mocks and shouldn't require a GPU. They primarily test the code structure and API.

### End-to-End Tests

Run the end-to-end tests with:

```bash
python src/test_cut_turbo_e2e.py
```

These tests require a GPU. You can selectively skip tests using command-line options:

```bash
python src/test_cut_turbo_e2e.py --skip-vae --skip-e2e  # Skip VAE and end-to-end tests
```

Available options:

- `--skip-vae`: Skip VAE component tests
- `--skip-patchnce`: Skip PatchNCE loss tests
- `--skip-patchsample`: Skip PatchSampleF tests
- `--skip-e2e`: Skip end-to-end model tests

## Expected Output

- Unit tests will report pass/fail status for each test case
- End-to-end tests will save images (`vae_input.png`, `vae_output.png`, `cut_input.png`, `cut_output.png`) for visual inspection and report metrics like reconstruction error and loss values

## Troubleshooting

1. **Memory Issues**: If you encounter CUDA out-of-memory errors, try:

   - Reducing batch sizes in the tests
   - Using smaller test images (modify `create_test_image` size parameter)
   - Running individual tests instead of the full suite

2. **Import Errors**: Make sure all dependencies are installed:

   ```bash
   pip install torch torchvision numpy pillow diffusers transformers peft
   ```

3. **Model Download Issues**: The tests download models from Hugging Face. If you have connection issues:
   - Use a VPN if your network blocks Hugging Face
   - Pre-download models to the Hugging Face cache directory
   - Consider using smaller models or creating minimal test models
