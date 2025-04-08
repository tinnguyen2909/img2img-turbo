import os
import gc
import copy
import lpips
import torch
import wandb
from glob import glob
import numpy as np
from accelerate import Accelerator
from accelerate.utils import set_seed
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, CLIPTextModel
from diffusers.optimization import get_scheduler
from peft.utils import get_peft_model_state_dict
from cleanfid.fid import get_folder_features, build_feature_extractor, frechet_distance
import vision_aided_loss
from model import make_1step_sched
from cut_turbo import CUT_Turbo, VAE_encode, VAE_decode, initialize_unet, initialize_vae
from my_utils.training_utils import UnpairedDataset, build_transform, parse_args_unpaired_training, UnpairedDataset_CutTurbo
from my_utils.dino_struct import DinoStructureLoss
from patch_nce import PatchSampleF, PatchNCELoss
import random


def main(args):
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, log_with=args.report_to)
    set_seed(args.seed)

    if accelerator.is_main_process:
        checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained("stabilityai/sd-turbo", subfolder="tokenizer", revision=args.revision, use_fast=False,)
    noise_scheduler_1step = make_1step_sched()
    text_encoder = CLIPTextModel.from_pretrained("stabilityai/sd-turbo", subfolder="text_encoder").cuda()

    unet, l_modules_unet_encoder, l_modules_unet_decoder, l_modules_unet_others = initialize_unet(args.lora_rank_unet, return_lora_module_names=True)
    vae_a2b, vae_lora_target_modules = initialize_vae(args.lora_rank_vae, return_lora_module_names=True)

    # Load checkpoint if continuing training
    patch_sample_f = initialize_patchnce(args)
    
    if args.continue_train and args.pretrained_model_name_or_path:
        if accelerator.is_main_process:
            print(f"Loading checkpoint from {args.pretrained_model_name_or_path}")
        
        # Create CUT_Turbo instance with the pretrained model
        model = CUT_Turbo(pretrained_path=args.pretrained_model_name_or_path)
        
        # Extract the components we need for training
        unet = model.unet
        vae_a2b = model.vae
        vae_b2a = model.vae # Only need one VAE in CUT
        vae_enc = model.vae_enc
        vae_dec = model.vae_dec
        
        # Load PatchSampleF if available in checkpoint
        checkpoint = torch.load(args.pretrained_model_name_or_path, map_location="cpu")
        if "patch_sample_f" in checkpoint:
            patch_sample_f.load_state_dict(checkpoint["patch_sample_f"])
            if accelerator.is_main_process:
                print("Successfully loaded PatchSampleF from checkpoint")
        
        # Verify that LoRA ranks match
        if checkpoint["rank_unet"] != args.lora_rank_unet:
            raise ValueError(f"Checkpoint LoRA rank ({checkpoint['rank_unet']}) does not match current setting ({args.lora_rank_unet})")
        if checkpoint["rank_vae"] != args.lora_rank_vae:
            raise ValueError(f"Checkpoint VAE LoRA rank ({checkpoint['rank_vae']}) does not match current setting ({args.lora_rank_vae})")
        
        if accelerator.is_main_process:
            print("Successfully loaded checkpoint")
    else:
        vae_b2a = copy.deepcopy(vae_a2b)  # We still need this for parameter initialization
        vae_enc = VAE_encode(vae_a2b)
        vae_dec = VAE_decode(vae_a2b)

    weight_dtype = torch.float32
    vae_a2b.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder.requires_grad_(False)

    # In CUT, we only need one discriminator for the target domain B
    if args.gan_disc_type == "vagan_clip":
        net_disc = vision_aided_loss.Discriminator(cv_type='clip', loss_type=args.gan_loss_type, device="cuda")
        net_disc.cv_ensemble.requires_grad_(False)  # Freeze feature extractor

    # Initialize PatchNCE components
    patch_sample_f.to(accelerator.device, dtype=weight_dtype)
    criterionNCE = PatchNCELoss(nce_temp=args.nce_temp, batch_size=args.train_batch_size) if args.lambda_NCE > 0 else None
    if criterionNCE is not None:
        criterionNCE.to(accelerator.device)

    crit_idt = torch.nn.L1Loss()

    if args.enable_xformers_memory_efficient_attention:
        unet.enable_xformers_memory_efficient_attention()

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    unet.conv_in.requires_grad_(True)
    params_gen = CUT_Turbo.get_traininable_params(unet, vae_a2b, vae_b2a, patch_sample_f)

    vae_enc = VAE_encode(vae_a2b)
    vae_dec = VAE_decode(vae_a2b)

    optimizer_gen = torch.optim.AdamW(params_gen, lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay, eps=args.adam_epsilon,)

    params_disc = list(net_disc.parameters())
    optimizer_disc = torch.optim.AdamW(params_disc, lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay, eps=args.adam_epsilon,)

    # dataset_train = UnpairedDataset(dataset_folder=args.dataset_folder, image_prep=args.train_img_prep, split="train", tokenizer=tokenizer)
    dataset_train = UnpairedDataset_CutTurbo(A=args.path_A, B=args.path_B, image_prep=args.train_img_prep, tokenizer=tokenizer, max_pairs=None)
    train_dataloader = torch.utils.data.DataLoader(dataset_train, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers)
    T_val = build_transform(args.val_img_prep)
    fixed_caption_src = dataset_train.fixed_caption_src
    fixed_caption_tgt = dataset_train.fixed_caption_tgt
    l_images_src_test = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
        for root, _, _ in os.walk(args.path_A):
            l_images_src_test.extend(glob(os.path.join(root, ext)))
    l_images_tgt_test = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
        for root, _, _ in os.walk(args.path_B):
            l_images_tgt_test.extend(glob(os.path.join(root, ext)))
    l_images_src_test, l_images_tgt_test = sorted(l_images_src_test), sorted(l_images_tgt_test)
    l_images_src_test = random.sample(l_images_src_test, 200)
    l_images_tgt_test = random.sample(l_images_tgt_test, 200)

    # make the reference FID statistics
    if accelerator.is_main_process:
        feat_model = build_feature_extractor("clean", "cuda", use_dataparallel=False)
        """
        FID reference statistics for A -> B translation
        """
        output_dir_ref = os.path.join(args.output_dir, "fid_reference_a2b")
        os.makedirs(output_dir_ref, exist_ok=True)
        # transform all images according to the validation transform and save them
        for _path in tqdm(l_images_tgt_test):
            _img = T_val(Image.open(_path).convert("RGB"))
            outf = os.path.join(output_dir_ref, os.path.basename(_path)).replace(".jpg", ".png")
            if not os.path.exists(outf):
                _img.save(outf)
        # compute the features for the reference images
        ref_features = get_folder_features(output_dir_ref, model=feat_model, num_workers=0, num=None,
                        shuffle=False, seed=0, batch_size=8, device=torch.device("cuda"),
                        mode="clean", custom_fn_resize=None, description="", verbose=True,
                        custom_image_tranform=None)
        a2b_ref_mu, a2b_ref_sigma = np.mean(ref_features, axis=0), np.cov(ref_features, rowvar=False)

    lr_scheduler_gen = get_scheduler(args.lr_scheduler, optimizer=optimizer_gen,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles, power=args.lr_power)
    lr_scheduler_disc = get_scheduler(args.lr_scheduler, optimizer=optimizer_disc,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles, power=args.lr_power)

    net_lpips = lpips.LPIPS(net='vgg')
    net_lpips.cuda()
    net_lpips.requires_grad_(False)

    fixed_a2b_tokens = tokenizer(fixed_caption_tgt, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids[0]
    fixed_a2b_emb_base = text_encoder(fixed_a2b_tokens.cuda().unsqueeze(0))[0].detach()
    del text_encoder, tokenizer  # free up some memory

    unet, vae_enc, vae_dec, net_disc, patch_sample_f, criterionNCE = accelerator.prepare(
        unet, vae_enc, vae_dec, net_disc, patch_sample_f, criterionNCE
    )
    net_lpips, optimizer_gen, optimizer_disc, train_dataloader, lr_scheduler_gen, lr_scheduler_disc = accelerator.prepare(
        net_lpips, optimizer_gen, optimizer_disc, train_dataloader, lr_scheduler_gen, lr_scheduler_disc
    )
    if accelerator.is_main_process:
        accelerator.init_trackers(args.tracker_project_name, config=dict(vars(args)))

    first_epoch = 0
    global_step = 0
    progress_bar = tqdm(range(0, args.max_train_steps), initial=global_step, desc="Steps",
        disable=not accelerator.is_local_main_process,)
    # turn off eff. attn for the disc
    for name, module in net_disc.named_modules():
        if "attn" in name:
            module.fused_attn = False

    for epoch in range(first_epoch, args.max_train_epochs):
        for step, batch in enumerate(train_dataloader):
            l_acc = [unet, net_disc, vae_enc, vae_dec, patch_sample_f]
            with accelerator.accumulate(*l_acc):
                img_a = batch["pixel_values_src"].to(dtype=weight_dtype)
                img_b = batch["pixel_values_tgt"].to(dtype=weight_dtype)

                bsz = img_a.shape[0]
                fixed_a2b_emb = fixed_a2b_emb_base.repeat(bsz, 1, 1).to(dtype=weight_dtype)
                timesteps = torch.tensor([noise_scheduler_1step.config.num_train_timesteps - 1] * bsz, device=img_a.device).long()

                """
                PatchNCE Objective
                """
                # Generate fake B from A with features using hooks
                fake_b, features_real_a = CUT_Turbo.forward_with_networks(
                    img_a, 
                    "a2b", 
                    vae_enc, 
                    unet, 
                    vae_dec, 
                    noise_scheduler_1step, 
                    timesteps, 
                    fixed_a2b_emb,
                    return_features=True
                )
                
                # Get features from fake B using the same hooks
                _, features_fake_b = CUT_Turbo.forward_with_networks(
                    fake_b,
                    "a2b",
                    vae_enc,
                    unet,
                    vae_dec,
                    noise_scheduler_1step,
                    timesteps,
                    fixed_a2b_emb,
                    return_features=True
                )
                
                # Calculate PatchNCE loss
                # Process features with correct shapes for PatchSampleF
                processed_features_real_a = []
                processed_features_fake_b = []
                
                for idx in range(len(features_real_a)):
                    feat_q = features_real_a[idx]
                    feat_k = features_fake_b[idx]
                    
                    # Make sure features are in correct shape (B, C, H, W)
                    if len(feat_q.shape) == 3:  # (B, L, C) -> (B, C, H, W)
                        h = w = int(np.sqrt(feat_q.shape[1]))
                        feat_q = feat_q.permute(0, 2, 1).reshape(feat_q.shape[0], feat_q.shape[2], h, w)
                        feat_k = feat_k.permute(0, 2, 1).reshape(feat_k.shape[0], feat_k.shape[2], h, w)
                    
                    # Ensure same device
                    feat_q = feat_q.to(device=accelerator.device)
                    feat_k = feat_k.to(device=accelerator.device)
                    
                    processed_features_real_a.append(feat_q)
                    processed_features_fake_b.append(feat_k)
                
                # Sample patches from all features at once
                feat_q_pool, patch_ids = patch_sample_f(processed_features_real_a, num_patches=args.num_patches, patch_ids=None)
                feat_k_pool, _ = patch_sample_f(processed_features_fake_b, num_patches=args.num_patches, patch_ids=patch_ids)
                
                # Calculate NCE loss across all features
                loss_nce = 0.0
                for q, k in zip(feat_q_pool, feat_k_pool):
                    loss_nce += criterionNCE(q, k).mean()
                
                loss_nce = loss_nce * args.lambda_NCE
                
                # Backward pass for NCE loss
                accelerator.backward(loss_nce, retain_graph=True)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_gen, args.max_grad_norm)
    
                optimizer_gen.step()
                lr_scheduler_gen.step()
                optimizer_gen.zero_grad()

                """
                Generator Objective (GAN) - only for A to B direction
                """
                # We already have fake_b from the PatchNCE step
                loss_gan = net_disc(fake_b, for_G=True).mean() * args.lambda_gan

                # Add new identity loss between real A and fake B (with detach)
                loss_idt_A = crit_idt(fake_b, img_a) * args.lambda_idt_A
                loss_idt_A += net_lpips(fake_b, img_a).mean() * args.lambda_idt_A_lpips
                loss_gan_with_idt = loss_gan + loss_idt_A
                accelerator.backward(loss_gan_with_idt, retain_graph=False)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_gen, args.max_grad_norm)
                optimizer_gen.step()
                lr_scheduler_gen.step()
                optimizer_gen.zero_grad()
                optimizer_disc.zero_grad()

                """
                Identity Objective - only for A to B direction
                """
                idt_b = CUT_Turbo.forward_with_networks(img_b, "a2b", vae_enc, unet, vae_dec, noise_scheduler_1step, timesteps, fixed_a2b_emb)
                loss_idt = crit_idt(idt_b, img_b) * args.lambda_idt
                loss_idt += net_lpips(idt_b, img_b).mean() * args.lambda_idt_lpips
                
                accelerator.backward(loss_idt, retain_graph=False)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_gen, args.max_grad_norm)
                optimizer_gen.step()
                lr_scheduler_gen.step()
                optimizer_gen.zero_grad()

                """
                Discriminator for fake B
                """
                loss_D_fake = net_disc(fake_b.detach(), for_real=False).mean() * args.lambda_gan
                accelerator.backward(loss_D_fake, retain_graph=False)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_disc, args.max_grad_norm)
                optimizer_disc.step()
                lr_scheduler_disc.step()
                optimizer_disc.zero_grad()

                """
                Discriminator for real B
                """
                loss_D_real = net_disc(img_b, for_real=True).mean() * args.lambda_gan
                accelerator.backward(loss_D_real, retain_graph=False)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_disc, args.max_grad_norm)
                optimizer_disc.step()
                lr_scheduler_disc.step()
                optimizer_disc.zero_grad()

            logs = {}
            logs["nce_loss"] = loss_nce.detach().item()
            logs["gan"] = loss_gan.detach().item()
            logs["disc"] = loss_D_fake.detach().item() + loss_D_real.detach().item()
            logs["idt"] = loss_idt.detach().item()
            logs["idt_A"] = loss_idt_A.detach().item()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    eval_unet = accelerator.unwrap_model(unet)
                    eval_vae_enc = accelerator.unwrap_model(vae_enc)
                    eval_vae_dec = accelerator.unwrap_model(vae_dec)
                    if global_step % args.viz_freq == 1:
                        for tracker in accelerator.trackers:
                            if tracker.name == "wandb":
                                viz_img_a = batch["pixel_values_src"].to(dtype=weight_dtype)
                                viz_img_b = batch["pixel_values_tgt"].to(dtype=weight_dtype)
                                log_dict = {
                                    "train/real_a": [wandb.Image(viz_img_a[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(bsz)],
                                    "train/real_b": [wandb.Image(viz_img_b[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(bsz)],
                                    "train/fake_b": [wandb.Image(fake_b[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(bsz)],
                                    "train/idt_b": [wandb.Image(idt_b[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(bsz)]
                                }
                                tracker.log(log_dict)
                                gc.collect()
                                torch.cuda.empty_cache()

                    if global_step % args.checkpointing_steps == 1:
                        # Save new checkpoint
                        outf = os.path.join(checkpoint_dir, f"model_{global_step}.pkl")
                        sd = {}
                        sd["l_target_modules_encoder"] = l_modules_unet_encoder
                        sd["l_target_modules_decoder"] = l_modules_unet_decoder
                        sd["l_modules_others"] = l_modules_unet_others
                        sd["rank_unet"] = args.lora_rank_unet
                        sd["sd_encoder"] = get_peft_model_state_dict(eval_unet, adapter_name="default_encoder")
                        sd["sd_decoder"] = get_peft_model_state_dict(eval_unet, adapter_name="default_decoder")
                        sd["sd_other"] = get_peft_model_state_dict(eval_unet, adapter_name="default_others")
                        sd["rank_vae"] = args.lora_rank_vae
                        sd["vae_lora_target_modules"] = vae_lora_target_modules
                        sd["sd_vae_enc"] = eval_vae_enc.state_dict()
                        sd["sd_vae_dec"] = eval_vae_dec.state_dict()
                        
                        # Save PatchSampleF state
                        eval_patch_sample_f = accelerator.unwrap_model(patch_sample_f)
                        sd["patch_sample_f"] = eval_patch_sample_f.state_dict()
                        
                        torch.save(sd, outf)
                        
                        # Keep only the 3 most recent checkpoints
                        if accelerator.is_main_process:
                            # Get all checkpoint files
                            checkpoint_files = glob(os.path.join(checkpoint_dir, "model_*.pkl"))
                            # Sort by modification time, newest first
                            checkpoint_files.sort(key=os.path.getmtime, reverse=True)
                            # Remove all but the 3 most recent
                            for old_checkpoint in checkpoint_files[3:]:
                                try:
                                    os.remove(old_checkpoint)
                                except Exception as e:
                                    print(f"Error removing old checkpoint {old_checkpoint}: {e}")
                        
                        gc.collect()
                        torch.cuda.empty_cache()

            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            if global_step >= args.max_train_steps:
                break


def initialize_patchnce(args):
    """Initialize the PatchSampleF network for PatchNCE loss"""
    patch_sample_f = PatchSampleF(use_mlp=True, nc=256, gpu_ids=[0])  # Use GPU
    return patch_sample_f


if __name__ == "__main__":
    args = parse_args_unpaired_training()
    # Add PatchNCE related arguments
    args.lambda_NCE = getattr(args, 'lambda_NCE', 1.0)  # Weight for PatchNCE loss
    args.nce_temp = getattr(args, 'nce_temp', 0.07)  # Temperature for NCE loss
    args.num_patches = getattr(args, 'num_patches', 256)  # Number of patches for NCE
    args.path_A = getattr(args, 'path_A', "data/A")
    args.path_B = getattr(args, 'path_B', "data/B")
    # Add new identity loss parameters
    args.lambda_idt_A = getattr(args, 'lambda_idt_A', 1.0)
    args.lambda_idt_A_lpips = getattr(args, 'lambda_idt_A_lpips', 1.0)
    # Add continue_train parameter
    args.continue_train = getattr(args, 'continue_train', False)
    main(args)
