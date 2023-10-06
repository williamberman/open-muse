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

    for i in range(5):
        generate_inpainting_images(model, vae, text_encoder, tokenizer, empty_embeds, empty_clip_embeds, i)

@torch.no_grad()
def generate_inpainting_images(
    model,
    vq_model,
    text_encoder,
    tokenizer,
    empty_embeds,
    empty_clip_embeds,
    n
):
    mask_token_id = model.config.vocab_size - 1

    validation_prompts, validation_images, validation_masks = inpainting_validation_data()

    validation_masks = validation_masks_to_latent_tensors(validation_masks).to(device)

    validation_images = torch.stack([TF.to_tensor(x) for x in validation_images])
    validation_images = validation_images.to(device)
    _, validation_images = vq_model.encode(validation_images)
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
    validation_masks = []

    for folder_name in os.listdir("./inpainting_validation_2"):
        validation_prompts.append(folder_name)

        image = None
        mask = None

        for file_name in os.listdir(f"./inpainting_validation_2/{folder_name}"):
            if file_name.startswith("image"):
                image = Image.open(f"./inpainting_validation_2/{folder_name}/{file_name}")

            if file_name.startswith("mask"):
                mask = Image.open(f"./inpainting_validation_2/{folder_name}/{file_name}").convert("L")

        assert image is not None, f"could not find inpainting validation image under {folder_name}"
        assert mask is not None, f"could not find inpainting validation mask under {folder_name}"

        validation_images.append(image)
        validation_masks.append(mask)

    return validation_prompts, validation_images, validation_masks


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