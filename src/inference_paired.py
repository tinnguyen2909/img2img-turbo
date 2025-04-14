import os
import argparse
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
from pix2pix_turbo import Pix2Pix_Turbo
from image_prep import canny_from_pil
from glob import glob

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', type=str, required=False, help='path to the input image')
    parser.add_argument('--input_dir', type=str, required=False, help='directory containing images to process')
    parser.add_argument('--max_images', type=int, default=None, help='maximum number of images to process from input_dir')
    parser.add_argument('--prompt', type=str, required=True, help='the prompt to be used')
    parser.add_argument('--model_name', type=str, default='', help='name of the pretrained model to be used')
    parser.add_argument('--model_path', type=str, default='', help='path to a model state dict to be used')
    parser.add_argument('--output_dir', type=str, default='output', help='the directory to save the output')
    parser.add_argument('--low_threshold', type=int, default=100, help='Canny low threshold')
    parser.add_argument('--high_threshold', type=int, default=200, help='Canny high threshold')
    parser.add_argument('--gamma', type=float, default=0.4, help='The sketch interpolation guidance amount')
    parser.add_argument('--seed', type=int, default=42, help='Random seed to be used')
    parser.add_argument('--use_fp16', action='store_true', help='Use Float16 precision for faster inference')
    args = parser.parse_args()

    # Ensure either input_image or input_dir is provided
    if args.input_image is None and args.input_dir is None:
        raise ValueError('Either --input_image or --input_dir must be provided')

    # only one of model_name and model_path should be provided
    if args.model_name == '' != args.model_path == '':
        raise ValueError('Either model_name or model_path should be provided')

    os.makedirs(args.output_dir, exist_ok=True)

    # initialize the model
    model = Pix2Pix_Turbo(pretrained_name=args.model_name, pretrained_path=args.model_path)
    model.set_eval()
    if args.use_fp16:
        model.half()

    # Process a single image
    def process_image(image_path):
        # make sure that the input image is a multiple of 8
        input_image = Image.open(image_path).convert('RGB')
        new_width = input_image.width - input_image.width % 8
        new_height = input_image.height - input_image.height % 8
        if new_width != input_image.width or new_height != input_image.height:
            input_image = input_image.resize((new_width, new_height), Image.LANCZOS)

        # Apply Canny edge detection
        canny_image = canny_from_pil(input_image, 
                                       args.low_threshold, 
                                       args.high_threshold).convert('RGB')
        canny_tensor = transforms.ToTensor()(canny_image).unsqueeze(0).cuda()
        if args.use_fp16:
            canny_tensor = canny_tensor.half()

        # Run through the model
        with torch.no_grad():
            # Invert the color for model (0: edge, 1: background)
            output = model(1.0 - canny_tensor, args.prompt, guidance_scale=args.gamma, seed=args.seed)

        # Save the output
        output_pil = transforms.ToPILImage()(output[0].cpu())
        
        # Save original and output
        bname = os.path.basename(image_path)
        output_pil.save(os.path.join(args.output_dir, f"generated_{bname}"))
        canny_image.save(os.path.join(args.output_dir, f"canny_{bname}"))
        return bname

    # Process images
    if args.input_image:
        # Process a single image
        bname = process_image(args.input_image)
        print(f"Processed: {args.input_image} -> {os.path.join(args.output_dir, f'generated_{bname}')}")
    else:
        # Process all images in the directory
        image_files = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
            image_files.extend(glob(os.path.join(args.input_dir, ext)))
        
        # Sort files for consistency
        image_files = sorted(image_files)
        
        # Limit the number of images if specified
        if args.max_images is not None:
            image_files = image_files[:args.max_images]
        
        # Process each image
        for image_path in image_files:
            bname = process_image(image_path)
            print(f"Processed: {image_path} -> {os.path.join(args.output_dir, f'generated_{bname}')}")
