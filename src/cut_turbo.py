import os
import sys
import copy
import torch
import torch.nn as nn
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
p = "src/"
sys.path.append(p)
from model import make_1step_sched, my_vae_encoder_fwd, my_vae_decoder_fwd, download_url
from patch_nce import PatchNCELoss, PatchSampleF


class UNetFeatureHook:
    def __init__(self):
        self.features = []
    
    def __call__(self, module, input, output):
        # For ResNet blocks: output is tuple (hidden_states,)
        # For CrossAttn blocks: output is tuple (hidden_states, res_samples)
        if isinstance(output, tuple):
            self.features.append(output[0].detach())
        else:
            self.features.append(output.detach())


class VAE_encode(nn.Module):
    def __init__(self, vae, vae_b2a=None):
        super(VAE_encode, self).__init__()
        self.vae = vae
        self.vae_b2a = vae_b2a

    def forward(self, x, direction):
        assert direction in ["a2b", "b2a"]
        if direction == "a2b":
            _vae = self.vae
        else:
            _vae = self.vae_b2a
        return _vae.encode(x).latent_dist.sample() * _vae.config.scaling_factor


class VAE_decode(nn.Module):
    def __init__(self, vae, vae_b2a=None):
        super(VAE_decode, self).__init__()
        self.vae = vae
        self.vae_b2a = vae_b2a

    def forward(self, x, direction):
        assert direction in ["a2b", "b2a"]
        if direction == "a2b":
            _vae = self.vae
        else:
            _vae = self.vae_b2a
        assert _vae.encoder.current_down_blocks is not None
        _vae.decoder.incoming_skip_acts = _vae.encoder.current_down_blocks
        x_decoded = (_vae.decode(x / _vae.config.scaling_factor).sample).clamp(-1, 1)
        return x_decoded


def initialize_unet(rank, return_lora_module_names=False):
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


class CUT_Turbo(torch.nn.Module):
    def __init__(self, pretrained_name=None, pretrained_path=None, ckpt_folder="checkpoints", lora_rank_unet=8, lora_rank_vae=4, nce_layers=None, nce_temp=0.07, num_patches=256):
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
        
        # Initialize PatchSampleF and PatchNCELoss for contrastive learning
        self.nce_layers = nce_layers if nce_layers is not None else [0, 4, 8, 12, 16]
        self.patch_sample_f = PatchSampleF(use_mlp=True, nc=256, gpu_ids=[0])
        self.criterionNCE = PatchNCELoss(nce_temp=nce_temp)
        self.num_patches = num_patches
        
        if pretrained_name == "day_to_night":
            url = "https://www.cs.cmu.edu/~img2img-turbo/models/day2night.pkl"
            self.load_ckpt_from_url(url, ckpt_folder)
            self.timesteps = torch.tensor([999], device="cuda").long()
            self.caption = "driving in the night"
            self.direction = "a2b"
        elif pretrained_name == "night_to_day":
            url = "https://www.cs.cmu.edu/~img2img-turbo/models/night2day.pkl"
            self.load_ckpt_from_url(url, ckpt_folder)
            self.timesteps = torch.tensor([999], device="cuda").long()
            self.caption = "driving in the day"
            self.direction = "b2a"
        elif pretrained_name == "clear_to_rainy":
            url = "https://www.cs.cmu.edu/~img2img-turbo/models/clear2rainy.pkl"
            self.load_ckpt_from_url(url, ckpt_folder)
            self.timesteps = torch.tensor([999], device="cuda").long()
            self.caption = "driving in heavy rain"
            self.direction = "a2b"
        elif pretrained_name == "rainy_to_clear":
            url = "https://www.cs.cmu.edu/~img2img-turbo/models/rainy2clear.pkl"
            self.load_ckpt_from_url(url, ckpt_folder)
            self.timesteps = torch.tensor([999], device="cuda").long()
            self.caption = "driving in the day"
            self.direction = "b2a"
        
        elif pretrained_path is not None:
            sd = torch.load(pretrained_path)
            self.load_ckpt_from_state_dict(sd)
            self.timesteps = torch.tensor([999], device="cuda").long()
            self.caption = None
            self.direction = None

        self.vae_enc.cuda()
        self.vae_dec.cuda()
        self.unet.cuda()
        self.patch_sample_f.cuda()

    def load_ckpt_from_state_dict(self, sd):
        # Check if adapters already exist and remove them if they do
        if hasattr(self.unet, "peft_config"):
            if "default_encoder" in self.unet.peft_config:
                self.unet.delete_adapter("default_encoder")
            if "default_decoder" in self.unet.peft_config:
                self.unet.delete_adapter("default_decoder")
            if "default_others" in self.unet.peft_config:
                self.unet.delete_adapter("default_others")
                
        # Now add the adapters
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

        # Check if VAE has adapter and remove it
        if hasattr(self.vae, "peft_config") and "vae_skip" in self.vae.peft_config:
            self.vae.delete_adapter("vae_skip")
            
        vae_lora_config = LoraConfig(r=sd["rank_vae"], init_lora_weights="gaussian", target_modules=sd["vae_lora_target_modules"])
        self.vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
        self.vae.decoder.gamma = 1
        # self.vae_b2a = copy.deepcopy(self.vae)
        self.vae_b2a = None
        self.vae_enc = VAE_encode(self.vae, vae_b2a=self.vae_b2a)
        self.vae_enc.load_state_dict(sd["sd_vae_enc"])
        self.vae_dec = VAE_decode(self.vae, vae_b2a=self.vae_b2a)
        self.vae_dec.load_state_dict(sd["sd_vae_dec"])
        
        # Load patch_sample_f state if present in the checkpoint
        if "patch_sample_f" in sd:
            # If there are MLPs in the state dict, make sure we initialize the MLPs first
            if any("mlp" in k for k in sd["patch_sample_f"].keys()):
                # Create dummy features to initialize MLPs
                dummy_features = [torch.randn(1, 256, 8, 8).cuda() for _ in range(3)]
                # This will trigger MLP creation
                self.patch_sample_f(dummy_features, num_patches=1) 
            # Now load the state dict
            self.patch_sample_f.load_state_dict(sd["patch_sample_f"])

    def load_ckpt_from_url(self, url, ckpt_folder):
        os.makedirs(ckpt_folder, exist_ok=True)
        outf = os.path.join(ckpt_folder, os.path.basename(url))
        download_url(url, outf)
        sd = torch.load(outf)
        self.load_ckpt_from_state_dict(sd)

    @staticmethod
    def forward_with_networks(x, direction, vae_enc, unet, vae_dec, sched, timesteps, text_emb, return_features=False):
        B = x.shape[0]
        assert direction in ["a2b", "b2a"]
        
        # Initialize hook containers if needed
        hook_handles = []
        feature_hook = None
        
        if return_features:
            feature_hook = UNetFeatureHook()
            
            # Register hooks to specific layers
            for block_idx, block in enumerate(unet.down_blocks):
                if hasattr(block, 'resnets'):
                    # Hook last ResNet layer in each down block
                    layer = block.resnets[-1]
                    handle = layer.register_forward_hook(feature_hook)
                    hook_handles.append(handle)
            
            # Hook mid block
            if hasattr(unet, 'mid_block'):
                handle = unet.mid_block.register_forward_hook(feature_hook)
                hook_handles.append(handle)
        
        x_enc = vae_enc(x, direction=direction).to(x.dtype)
        model_pred = unet(x_enc, timesteps, encoder_hidden_states=text_emb,).sample
        
        # Remove hooks immediately after use
        if return_features:
            for handle in hook_handles:
                handle.remove()
        
        x_out = torch.stack([sched.step(model_pred[i], timesteps[i], x_enc[i], return_dict=True).prev_sample for i in range(B)])
        x_out_decoded = vae_dec(x_out, direction=direction)
        
        if return_features:
            return x_out_decoded, feature_hook.features
        else:
            return x_out_decoded

    @staticmethod
    def get_traininable_params(unet, vae_a2b, vae_b2a, patch_sample_f=None):
        # Use a set to track already added parameter tensors to avoid duplicates
        param_set = set()
        params_gen = []
        
        # Add all unet parameters
        unet.conv_in.requires_grad_(True)
        unet.set_adapters(["default_encoder", "default_decoder", "default_others"])
        
        # Add conv_in parameters 
        for p in unet.conv_in.parameters():
            if p.requires_grad and p not in param_set:
                params_gen.append(p)
                param_set.add(p)
        
        # Add LoRA parameters
        for n, p in unet.named_parameters():
            if "lora" in n and "default" in n:
                assert p.requires_grad
                if p not in param_set:
                    params_gen.append(p)
                    param_set.add(p)
        
        # Add all vae_a2b parameters
        for n, p in vae_a2b.named_parameters():
            if "lora" in n and "vae_skip" in n and p.requires_grad and p not in param_set:
                params_gen.append(p)
                param_set.add(p)
        
        # Add skip connection parameters for vae_a2b
        for param in vae_a2b.decoder.skip_conv_1.parameters():
            if param not in param_set:
                params_gen.append(param)
                param_set.add(param)
                
        for param in vae_a2b.decoder.skip_conv_2.parameters():
            if param not in param_set:
                params_gen.append(param)
                param_set.add(param)
                
        for param in vae_a2b.decoder.skip_conv_3.parameters():
            if param not in param_set:
                params_gen.append(param)
                param_set.add(param)
                
        for param in vae_a2b.decoder.skip_conv_4.parameters():
            if param not in param_set:
                params_gen.append(param)
                param_set.add(param)

        # Add all vae_b2a parameters (only for initialization; not used in CUT)
        # for n, p in vae_b2a.named_parameters():
        #     if "lora" in n and "vae_skip" in n and p.requires_grad and p not in param_set:
        #         params_gen.append(p)
        #         param_set.add(p)
                
        # for param in vae_b2a.decoder.skip_conv_1.parameters():
        #     if param not in param_set:
        #         params_gen.append(param)
        #         param_set.add(param)
                
        # for param in vae_b2a.decoder.skip_conv_2.parameters():
        #     if param not in param_set:
        #         params_gen.append(param)
        #         param_set.add(param)
                
        # for param in vae_b2a.decoder.skip_conv_3.parameters():
        #     if param not in param_set:
        #         params_gen.append(param)
        #         param_set.add(param)
                
        # for param in vae_b2a.decoder.skip_conv_4.parameters():
        #     if param not in param_set:
        #         params_gen.append(param)
        #         param_set.add(param)
        
        # Add patch_sample_f parameters for contrastive learning
        if patch_sample_f is not None:
            for param in patch_sample_f.parameters():
                if param not in param_set:
                    params_gen.append(param)
                    param_set.add(param)
            
        return params_gen

    def forward(self, x_t, direction=None, caption=None, caption_emb=None, return_features=False):
        if direction is None:
            assert self.direction is not None
            direction = self.direction
        if caption is None and caption_emb is None:
            assert self.caption is not None
            caption = self.caption
        if caption_emb is not None:
            caption_enc = caption_emb
        else:
            caption_tokens = self.tokenizer(caption, max_length=self.tokenizer.model_max_length,
                    padding="max_length", truncation=True, return_tensors="pt").input_ids.to(x_t.device)
            caption_enc = self.text_encoder(caption_tokens)[0].detach().clone()
        return self.forward_with_networks(x_t, direction, self.vae_enc, self.unet, self.vae_dec, self.sched, self.timesteps, caption_enc, return_features)
