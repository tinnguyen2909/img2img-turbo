import os
import argparse
from PIL import Image
import torch
from torchvision import transforms
import clip
from glob import glob

from pix2pix_turbo import Pix2Pix_Turbo
from my_utils.training_utils import build_transform


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', type=str, required=False, help='path to the input source image')
    parser.add_argument('--input_dir', type=str, required=False, help='directory containing source images to process')
    parser.add_argument('--model_path', type=str, required=True, help='path to the Pix2Pix_Turbo model checkpoint (.pkl)')
    parser.add_argument('--prompt', type=str, required=True, help='the prompt to guide the image translation')
    parser.add_argument('--output_dir', type=str, default='output_paired', help='the directory to save the output images')
    parser.add_argument('--image_prep', type=str, default='resize_512x512', help='the image preparation method (e.g., resize_512x512)')
    parser.add_argument('--use_fp16', action='store_true', help='Use Float16 precision for faster inference')
    parser.add_argument('--max_images', type=int, default=None, help='maximum number of images to process from input_dir')
    args = parser.parse_args()

    # Ensure either input_image or input_dir is provided
    if args.input_image is None and args.input_dir is None:
        raise ValueError('Either --input_image or --input_dir must be provided')
    if args.input_image is not None and args.input_dir is not None:
        raise ValueError('Provide either --input_image or --input_dir, not both')

    # Initialize the model
    model = Pix2Pix_Turbo(pretrained_path=args.model_path)
    model.eval()
    model.unet.enable_xformers_memory_efficient_attention()
    if args.use_fp16:
        model.half()
    model.cuda()

    T_val = build_transform(args.image_prep)
    os.makedirs(args.output_dir, exist_ok=True)

    # Prepare prompt tokens
    prompt_tokens = clip.tokenize(args.prompt, truncate=True).cuda()

    # Process a single image
    def process_image(image_path):
        input_image = Image.open(image_path).convert('RGB')
        # Prepare input
        with torch.no_grad():
            source_img_pil = T_val(input_image)
            source_img_tensor = transforms.ToTensor()(source_img_pil).unsqueeze(0).cuda()
            # Input should be [0, 1] as per MyPairedDataset conditioning_pixel_values
            # source_img_tensor = transforms.Normalize([0.5], [0.5])(source_img_tensor).unsqueeze(0).cuda()

            if args.use_fp16:
                source_img_tensor = source_img_tensor.half()

            # Run inference
            output = model(source_img_tensor, prompt_tokens=prompt_tokens, deterministic=True)

        output_pil = transforms.ToPILImage()(output[0].cpu().float() * 0.5 + 0.5)

        # Save the output image
        bname = os.path.basename(image_path)
        output_path = os.path.join(args.output_dir, bname)
        output_pil.save(output_path)
        return bname, output_path

    # Determine input source
    if args.input_image:
        image_files = [args.input_image]
    else:
        image_files = []
        supported_exts = ["*.jpg", "*.jpeg", "*.png", "*.webp"]
        for ext in supported_exts:
            # Search recursively
            image_files.extend(glob(os.path.join(args.input_dir, f"**/{ext}"), recursive=True))
        
        # Sort files for consistency
        image_files = sorted(list(set(image_files))) # Use set to remove duplicates if ** matches parent dirs

        # Limit the number of images if specified
        if args.max_images is not None:
            image_files = image_files[:args.max_images]

    # Process images
    total_images = len(image_files)
    if total_images == 0:
        print(f"No images found in {args.input_dir} with extensions {supported_exts}")
    else:
        print(f"Starting processing of {total_images} images...")
        for i, image_path in enumerate(image_files):
            try:
                bname, output_path = process_image(image_path)
                print(f"Processed [{i+1}/{total_images}]: {image_path} -> {output_path}")
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue

    print("Inference complete.")
