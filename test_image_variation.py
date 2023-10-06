import torch
import os
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
from muse import MaskGiTUViT, VQGANModel
from transformers import CLIPTextModelWithProjection, CLIPTokenizer
    
device = 'cuda'

def main():
    model = MaskGiTUViT.from_pretrained('../research-run-512-with-downsample-checkpoint-490000')
    model.to(device)
    text_encoder = CLIPTextModelWithProjection.from_pretrained("openMUSE/clip-vit-large-patch14-text-enc")
    text_encoder.to(device)
    tokenizer = CLIPTokenizer.from_pretrained("openMUSE/clip-vit-large-patch14-text-enc")
    vae = VQGANModel.from_pretrained("openMUSE/vqgan-f16-8192-laion")
    vae.to(device)

    empty_input = tokenizer("", padding="max_length", return_tensors="pt").input_ids.to(device)
    outputs = text_encoder(empty_input, output_hidden_states=True)
    empty_embeds = outputs.hidden_states[-2]
    empty_clip_embeds = outputs[0]

    mask_ratio = 0.75

    for i in range(5):
        generate_inpainting_images(model, vae, text_encoder, tokenizer, empty_embeds, empty_clip_embeds, i, mask_ratio)

@torch.no_grad()
def generate_inpainting_images(
    model,
    vq_model,
    text_encoder,
    tokenizer,
    empty_embeds,
    empty_clip_embeds,
    n,
    mask_ratio
):
    mask_token_id = model.config.vocab_size - 1

    validation_prompts, validation_images, mask_images = inpainting_validation_data()

    validation_images = torch.stack([TF.to_tensor(x) for x in validation_images])
    validation_images = validation_images.to(device)
    _, validation_images = vq_model.encode(validation_images)
    validation_masks = torch.rand(validation_images.shape, device=validation_images.device) < mask_ratio
    validation_images[validation_masks] = mask_token_id

    token_input_ids = tokenizer(
        validation_prompts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=77,
    ).input_ids

    outputs = text_encoder(token_input_ids.to(device), return_dict=True, output_hidden_states=True)
    encoder_hidden_states = outputs.hidden_states[-2]
    clip_embeds = outputs[0]

    resolution = 512
    micro_conds = torch.tensor(
        [resolution, resolution, 0, 0, 6], device=encoder_hidden_states.device, dtype=encoder_hidden_states.dtype
    )
    micro_conds = micro_conds.unsqueeze(0).repeat(encoder_hidden_states.shape[0], 1)

    with torch.autocast("cuda", dtype=torch.float16):
        # Generate images
        gen_token_ids = model.generate2(
            input_ids=validation_images,
            encoder_hidden_states=encoder_hidden_states,
            cond_embeds=clip_embeds,
            empty_embeds=empty_embeds,
            empty_cond_embeds=empty_clip_embeds,
            micro_conds=micro_conds,
            guidance_scale=8,
        )
    # In the beginning of training, the model is not fully trained and the generated token ids can be out of range
    # so we clamp them to the correct range.
    gen_token_ids = torch.clamp(gen_token_ids, max=model.config.codebook_size - 1)

    mask_images = validation_masks_to_latent_tensors(mask_images)
    gen_token_ids[mask_images] = mask_token_id

    with torch.autocast("cuda", dtype=torch.float16):
        # Generate images
        gen_token_ids = model.generate2(
            input_ids=validation_images,
            encoder_hidden_states=encoder_hidden_states,
            cond_embeds=clip_embeds,
            empty_embeds=empty_embeds,
            empty_cond_embeds=empty_clip_embeds,
            micro_conds=micro_conds,
            guidance_scale=8,
            timesteps=25,
        )

    gen_token_ids = torch.clamp(gen_token_ids, max=model.config.codebook_size - 1)

    images = vq_model.decode_code(gen_token_ids)

    # Convert to PIL images
    images = 2.0 * images - 1.0
    images = torch.clamp(images, -1.0, 1.0)
    images = (images + 1.0) / 2.0
    images *= 255.0
    images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    pil_images = [Image.fromarray(image) for image in images]

    for i, image in enumerate(pil_images):
        image.save(f'./out-{n}-{i}.png')


def inpainting_validation_data():
    validation_prompts = []
    validation_images = []
    mask_images = []

    validation_prompts.append(
        "a dog"
    )

    validation_images.append(
        Image.open('./cat_real.png').convert("RGB")
    )

    mask_images.append(
        Image.open('./cat_real_mask.png').convert('L')
    )

    validation_prompts.append(
        "a cow"
    )

    validation_images.append(
        Image.open('./cat_real.png').convert("RGB")
    )

    mask_images.append(
        Image.open('./cat_real_mask.png').convert('L')
    )

    return validation_prompts, validation_images, mask_images


def validation_masks_to_latent_tensors(validation_masks):
    validation_masks_ = []

    for mask in validation_masks:
        mask = mask.resize((mask.height // 16, mask.width // 16))
        mask = np.array(mask)
        mask = mask / 255
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        mask = mask.reshape(-1)
        mask = mask.astype(bool)
        validation_masks_.append(mask)

    validation_masks_ = np.stack(validation_masks_)

    return torch.from_numpy(validation_masks_)

if __name__ == "__main__":
    main()