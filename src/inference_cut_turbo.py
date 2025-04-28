import os
import argparse
from PIL import Image
import torch
from torchvision import transforms
from cut_turbo import CUT_Turbo
from my_utils.training_utils import build_transform
from glob import glob


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', type=str, required=False, help='path to the input image')
    parser.add_argument('--input_dir', type=str, required=False, help='directory containing images to process')
    parser.add_argument('--max_images', type=int, default=None, help='maximum number of images to process from input_dir')
    parser.add_argument('--prompt', type=str, required=False, help='the prompt to be used. It is required when loading a custom model_path.')
    parser.add_argument('--model_name', type=str, default=None, help='name of the pretrained model to be used')
    parser.add_argument('--model_path', type=str, default=None, help='path to a local model state dict to be used')
    parser.add_argument('--output_dir', type=str, default='output', help='the directory to save the output')
    parser.add_argument('--image_prep', type=str, default='resize_512x512', help='the image preparation method')
    parser.add_argument('--direction', type=str, default=None, help='the direction of translation. None for pretrained models, a2b or b2a for custom paths.')
    parser.add_argument('--use_fp16', action='store_true', help='Use Float16 precision for faster inference')
    args = parser.parse_args()

    # Ensure either input_image or input_dir is provided
    if args.input_image is None and args.input_dir is None:
        raise ValueError('Either --input_image or --input_dir must be provided')

    # Only one of model_name and model_path should be provided
    if args.model_name is None != args.model_path is None:
        raise ValueError('Either model_name or model_path should be provided')

    if args.model_path is not None and args.prompt is None:
        raise ValueError('prompt is required when loading a custom model_path.')

    if args.model_name is not None:
        assert args.prompt is None, 'prompt is not required when loading a pretrained model.'
        assert args.direction is None, 'direction is not required when loading a pretrained model.'

    # Initialize the model
    model = CUT_Turbo(pretrained_name=args.model_name, pretrained_path=args.model_path)
    model.eval()
    model.unet.enable_xformers_memory_efficient_attention()
    if args.use_fp16:
        model.half()

    T_val = build_transform(args.image_prep)
    os.makedirs(args.output_dir, exist_ok=True)

    # Process a single image
    def process_image(image_path):
        input_image = Image.open(image_path).convert('RGB')
        # Translate the image
        with torch.no_grad():
            input_img = T_val(input_image)
            x_t = transforms.ToTensor()(input_img)
            x_t = transforms.Normalize([0.5], [0.5])(x_t).unsqueeze(0).cuda()
            if args.use_fp16:
                x_t = x_t.half()
            output = model(x_t, direction=args.direction, caption=args.prompt)

        output_pil = transforms.ToPILImage()(output[0].cpu() * 0.5 + 0.5)
        
        # Save the output image
        bname = os.path.basename(image_path)
        output_pil.save(os.path.join(args.output_dir, bname))
        return bname
    
    # Process images
    if args.input_image:
        # Process a single image
        process_image(args.input_image)
        print(f"Processed: {args.input_image}")
    else:
        # Process all images in the directory and its subdirectories
        image_files = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
            image_files.extend(glob(os.path.join(args.input_dir, f"**/{ext}"), recursive=True))
        
        # Sort files for consistency
        image_files = sorted(image_files)
        
        # Limit the number of images if specified
        if args.max_images is not None:
            image_files = image_files[:args.max_images]
        
        total_images = len(image_files)
        print(f"Found {total_images} images to process")
        
        # Process each image
        for i, image_path in enumerate(image_files, 1):
            bname = process_image(image_path)
            print(f"[{i}/{total_images}] Processed: {image_path} -> {os.path.join(args.output_dir, bname)}")
        
        print(f"Completed processing {total_images} images") 