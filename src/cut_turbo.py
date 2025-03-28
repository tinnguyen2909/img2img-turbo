import os
import sys
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
p = "src/"
sys.path.append(p)
from model import make_1step_sched, my_vae_encoder_fwd, my_vae_decoder_fwd, download_url


class VAE_encode(nn.Module):
    def __init__(self, vae):
        super(VAE_encode, self).__init__()
        self.vae = vae

    def forward(self, x):
        return self.vae.encode(x).latent_dist.sample() * self.vae.config.scaling_factor


class VAE_decode(nn.Module):
    def __init__(self, vae):
        super(VAE_decode, self).__init__()
        self.vae = vae

    def forward(self, x):
        assert self.vae.encoder.current_down_blocks is not None
        self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks
        x_decoded = (self.vae.decode(x / self.vae.config.scaling_factor).sample).clamp(-1, 1)
        return x_decoded


def initialize_unet(rank, return_lora_module_names=False):
    """Initialize UNet with LoRA adapters for the CUT model"""
    unet = UNet2DConditionModel.from_pretrained("stabilityai/sd-turbo", subfolder="unet")
    unet.requires_grad_(False)
    unet.train()
    l_target_modules_encoder, l_target_modules_decoder, l_modules_others = [], [], []
    l_grep = ["to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_in", "conv_shortcut", "conv_out", "proj_out", "proj_in", "ff.net.2", "ff.net.0.proj"]
    for n, p in unet.named_parameters():
        if "bias" in n or "norm" in n: continue
        for pattern in l_grep:
            if pattern in n and ("down_blocks" in n or "conv_in" in n):
                l_target_modules_encoder.append(n.replace(".weight",""))
                break
            elif pattern in n and "up_blocks" in n:
                l_target_modules_decoder.append(n.replace(".weight",""))
                break
            elif pattern in n:
                l_modules_others.append(n.replace(".weight",""))
                break
    lora_conf_encoder = LoraConfig(r=rank, init_lora_weights="gaussian",target_modules=l_target_modules_encoder, lora_alpha=rank)
    lora_conf_decoder = LoraConfig(r=rank, init_lora_weights="gaussian",target_modules=l_target_modules_decoder, lora_alpha=rank)
    lora_conf_others = LoraConfig(r=rank, init_lora_weights="gaussian",target_modules=l_modules_others, lora_alpha=rank)
    unet.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
    unet.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
    unet.add_adapter(lora_conf_others, adapter_name="default_others")
    unet.set_adapters(["default_encoder", "default_decoder", "default_others"])
    if return_lora_module_names:
        return unet, l_target_modules_encoder, l_target_modules_decoder, l_modules_others
    else:
        return unet


def initialize_vae(rank=4, return_lora_module_names=False):
    """Initialize VAE with LoRA for the CUT model"""
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-turbo", subfolder="vae")
    vae.requires_grad_(False)
    vae.encoder.forward = my_vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)
    vae.decoder.forward = my_vae_decoder_fwd.__get__(vae.decoder, vae.decoder.__class__)
    vae.requires_grad_(True)
    vae.train()
    # add the skip connection convs
    vae.decoder.skip_conv_1 = torch.nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda().requires_grad_(True)
    vae.decoder.skip_conv_2 = torch.nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda().requires_grad_(True)
    vae.decoder.skip_conv_3 = torch.nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda().requires_grad_(True)
    vae.decoder.skip_conv_4 = torch.nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda().requires_grad_(True)
    torch.nn.init.constant_(vae.decoder.skip_conv_1.weight, 1e-5)
    torch.nn.init.constant_(vae.decoder.skip_conv_2.weight, 1e-5)
    torch.nn.init.constant_(vae.decoder.skip_conv_3.weight, 1e-5)
    torch.nn.init.constant_(vae.decoder.skip_conv_4.weight, 1e-5)
    vae.decoder.ignore_skip = False
    vae.decoder.gamma = 1
    l_vae_target_modules = ["conv1","conv2","conv_in", "conv_shortcut",
        "conv", "conv_out", "skip_conv_1", "skip_conv_2", "skip_conv_3", 
        "skip_conv_4", "to_k", "to_q", "to_v", "to_out.0",
    ]
    vae_lora_config = LoraConfig(r=rank, init_lora_weights="gaussian", target_modules=l_vae_target_modules)
    vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
    if return_lora_module_names:
        return vae, l_vae_target_modules
    else:
        return vae


class PatchNCELoss(nn.Module):
    """PatchNCE loss for contrastive learning between corresponding patches"""
    def __init__(self, nce_layers, nce_temp=0.07, nce_includes_all_negatives_from_minibatch=False):
        super().__init__()
        self.nce_layers = nce_layers
        self.nce_temp = nce_temp
        self.nce_includes_all_negatives_from_minibatch = nce_includes_all_negatives_from_minibatch
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.bool

    def forward(self, feat_q, feat_k):
        batch_size = feat_q[0].shape[0]
        dim = feat_q[0].shape[1]
        total_loss = 0.0
        
        for f_q, f_k in zip(feat_q, feat_k):
            n_patches = f_q.shape[2] * f_q.shape[3]
            f_q = f_q.permute(0, 2, 3, 1).reshape(batch_size, n_patches, dim)
            f_k = f_k.permute(0, 2, 3, 1).reshape(batch_size, n_patches, dim)
            
            # Calculate L_NCE for each layer
            l_pos = torch.bmm(f_q, f_k.transpose(1, 2)) / self.nce_temp
            
            # For each query patch, use all other patches as negatives
            if self.nce_includes_all_negatives_from_minibatch:
                # Reshape to include all patches from all batch elements
                f_q_all = f_q.reshape(-1, dim)
                f_k_all = f_k.reshape(-1, dim)
                l_neg_curbatch = torch.mm(f_q_all, f_k_all.t()) / self.nce_temp
                
                # Create a positive mask for the diagonal (corresponding patches)
                diagonal = torch.eye(batch_size * n_patches, device=f_q.device, dtype=self.mask_dtype)
                l_neg_curbatch.masked_fill_(diagonal, -10.0)
                l_neg = l_neg_curbatch.reshape(batch_size, n_patches, -1)
            else:
                # For each query patch, use all non-corresponding patches from the same image as negatives
                identity_matrix = torch.eye(n_patches, device=f_q.device, dtype=self.mask_dtype)[None, :, :]
                l_neg = torch.bmm(f_q, f_q.transpose(1, 2)) / self.nce_temp
                l_neg.masked_fill_(identity_matrix, -10.0)  # Mask out positive examples
            
            # Concatenate positive and negative logits and create labels
            out = torch.cat((l_pos, l_neg), dim=2)
            logits = out.reshape(-1, out.size(2))
            labels = torch.zeros(logits.size(0), dtype=torch.long, device=f_q.device)
            
            loss = self.cross_entropy_loss(logits, labels)
            loss = loss.reshape(batch_size, n_patches).mean()
            total_loss += loss
            
        return total_loss / len(self.nce_layers)


class PatchSampleF(nn.Module):
    """PatchSample module that samples features from different layers"""
    def __init__(self, nce_layers, use_mlp=True, nc=256):
        super().__init__()
        self.nce_layers = nce_layers
        self.use_mlp = use_mlp
        if use_mlp:
            self.mlps = nn.ModuleList()
            for layer_id in range(len(nce_layers)):
                input_nc = 512  # Adjust based on UNet block output dimensions
                mlp = nn.Sequential(
                    nn.Conv2d(input_nc, nc, kernel_size=1, padding=0, bias=True),
                    nn.ReLU(),
                    nn.Conv2d(nc, nc, kernel_size=1, padding=0, bias=True),
                )
                self.mlps.append(mlp)

    def forward(self, unet, z, text_emb, timesteps, layers=None):
        if layers is None:
            layers = self.nce_layers
            
        features = []
        result = unet(z, timesteps, encoder_hidden_states=text_emb, output_hidden_states=True)
        hidden_states = result.hidden_states
        
        for layer_id in layers:
            feat = hidden_states[layer_id]
            if self.use_mlp:
                feat = self.mlps[layer_id](feat)
            features.append(feat)
            
        return features


class CUT_Turbo(torch.nn.Module):
    def __init__(self, pretrained_name=None, pretrained_path=None, ckpt_folder="checkpoints", lora_rank_unet=8, lora_rank_vae=4):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("stabilityai/sd-turbo", subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained("stabilityai/sd-turbo", subfolder="text_encoder").cuda()
        self.sched = make_1step_sched()
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-turbo", subfolder="vae")
        unet = UNet2DConditionModel.from_pretrained("stabilityai/sd-turbo", subfolder="unet")
        vae.encoder.forward = my_vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)
        vae.decoder.forward = my_vae_decoder_fwd.__get__(vae.decoder, vae.decoder.__class__)
        
        # add the skip connection convs
        vae.decoder.skip_conv_1 = torch.nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.skip_conv_2 = torch.nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.skip_conv_3 = torch.nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.skip_conv_4 = torch.nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.ignore_skip = False
        
        self.unet, self.vae = unet, vae
        
        # Define NCE layers - these represent which UNet layers to use for PatchNCE loss
        self.nce_layers = [4, 8, 12]  # Example layers from UNet hidden states
        
        # Initialize PatchSampleF module for feature sampling
        self.patch_sample = PatchSampleF(self.nce_layers, use_mlp=True)
        
        # Initialize PatchNCE loss
        self.criterionNCE = PatchNCELoss(self.nce_layers)
        
        if pretrained_path is not None:
            sd = torch.load(pretrained_path)
            self.load_ckpt_from_state_dict(sd)
            self.timesteps = torch.tensor([999], device="cuda").long()
            self.caption = None
            self.direction = None

        self.vae_enc = VAE_encode(self.vae)
        self.vae_dec = VAE_decode(self.vae)
        self.vae_enc.cuda()
        self.vae_dec.cuda()
        self.unet.cuda()

    def load_ckpt_from_state_dict(self, sd):
        lora_conf_encoder = LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian", target_modules=sd["l_target_modules_encoder"], lora_alpha=sd["rank_unet"])
        lora_conf_decoder = LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian", target_modules=sd["l_target_modules_decoder"], lora_alpha=sd["rank_unet"])
        lora_conf_others = LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian", target_modules=sd["l_modules_others"], lora_alpha=sd["rank_unet"])
        self.unet.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
        self.unet.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
        self.unet.add_adapter(lora_conf_others, adapter_name="default_others")
        for n, p in self.unet.named_parameters():
            name_sd = n.replace(".default_encoder.weight", ".weight")
            if "lora" in n and "default_encoder" in n:
                p.data.copy_(sd["sd_encoder"][name_sd])
        for n, p in self.unet.named_parameters():
            name_sd = n.replace(".default_decoder.weight", ".weight")
            if "lora" in n and "default_decoder" in n:
                p.data.copy_(sd["sd_decoder"][name_sd])
        for n, p in self.unet.named_parameters():
            name_sd = n.replace(".default_others.weight", ".weight")
            if "lora" in n and "default_others" in n:
                p.data.copy_(sd["sd_other"][name_sd])
        self.unet.set_adapter(["default_encoder", "default_decoder", "default_others"])

        vae_lora_config = LoraConfig(r=sd["rank_vae"], init_lora_weights="gaussian", target_modules=sd["vae_lora_target_modules"])
        self.vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
        self.vae.decoder.gamma = 1
        
        # Load patch sample network parameters if they exist
        if "sd_patch_sample" in sd:
            self.patch_sample.load_state_dict(sd["sd_patch_sample"])

    def load_ckpt_from_url(self, url, ckpt_folder):
        os.makedirs(ckpt_folder, exist_ok=True)
        outf = os.path.join(ckpt_folder, os.path.basename(url))
        download_url(url, outf)
        sd = torch.load(outf)
        self.load_ckpt_from_state_dict(sd)

    @staticmethod
    def forward_generator(img_src, vae_enc, unet, vae_dec, sched, timesteps, text_emb):
        """Forward pass for the generator part (source → target)"""
        B = img_src.shape[0]
        x_enc = vae_enc(img_src).to(img_src.dtype)
        model_pred = unet(x_enc, timesteps, encoder_hidden_states=text_emb,).sample
        x_out = torch.stack([sched.step(model_pred[i], timesteps[i], x_enc[i], return_dict=True).prev_sample for i in range(B)])
        x_out_decoded = vae_dec(x_out)
        return x_out_decoded, x_enc

    @staticmethod
    def compute_nce_loss(real_src, fake_tgt, enc_src, unet, patch_sample, criterionNCE, timesteps, text_emb):
        """Compute PatchNCE loss between real source images and generated target images"""
        # Get features from real source images
        src_features = patch_sample(unet, enc_src, text_emb, timesteps)
        
        # Re-encode fake target images for feature comparison
        with torch.no_grad():
            enc_fake = unet(enc_src, timesteps, encoder_hidden_states=text_emb, output_hidden_states=True).hidden_states
            fake_features = [feat.detach() for feat in enc_fake]
        
        # Calculate PatchNCE loss
        loss_NCE = criterionNCE(src_features, fake_features)
        
        return loss_NCE

    @staticmethod
    def get_trainable_params(unet, vae, patch_sample):
        """Get all trainable parameters for the CUT model"""
        # UNet parameters
        params = list(unet.conv_in.parameters())
        unet.conv_in.requires_grad_(True)
        unet.set_adapters(["default_encoder", "default_decoder", "default_others"])
        for n, p in unet.named_parameters():
            if "lora" in n and "default" in n:
                assert p.requires_grad
                params.append(p)
        
        # VAE parameters
        for n, p in vae.named_parameters():
            if "lora" in n and "vae_skip" in n:
                assert p.requires_grad
                params.append(p)
        params = params + list(vae.decoder.skip_conv_1.parameters())
        params = params + list(vae.decoder.skip_conv_2.parameters())
        params = params + list(vae.decoder.skip_conv_3.parameters())
        params = params + list(vae.decoder.skip_conv_4.parameters())
        
        # PatchSampleF parameters
        params = params + list(patch_sample.parameters())
        
        return params

    def forward(self, x_t, caption=None, caption_emb=None):
        if caption is None and caption_emb is None:
            raise ValueError("Either caption or caption_emb must be provided")
        
        if caption_emb is not None:
            caption_enc = caption_emb
        else:
            caption_tokens = self.tokenizer(
                caption, max_length=self.tokenizer.model_max_length,
                padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids.cuda()
            caption_enc = self.text_encoder(caption_tokens)[0]
        
        x_enc = self.vae_enc(x_t)
        model_pred = self.unet(x_enc, self.timesteps, encoder_hidden_states=caption_enc,).sample
        x_out = self.sched.step(model_pred, self.timesteps, x_enc, return_dict=True).prev_sample
        x_out_decoded = self.vae_dec(x_out)
        
        return x_out_decoded 