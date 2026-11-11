import os
from types import SimpleNamespace
from typing import Optional, Tuple, List

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from clip.model import CLIP
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import CLIPImageProcessor, CLIPTextModelWithProjection, CLIPVisionModelWithProjection

from data_utils import collate_fn
from phi import Phi

if torch.cuda.is_available():
    device = torch.device("cuda")
    dtype = torch.float16
else:
    device = torch.device("cpu")
    dtype = torch.float32


class HFCLIPWrapper(nn.Module):
    """
    Wrapper giúp đồng nhất giao diện của Hugging Face CLIP (LinCIR) với OpenAI CLIP (CIG).
    """
    def __init__(self, image_encoder: CLIPVisionModelWithProjection, text_encoder: CLIPTextModelWithProjection):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.is_hf = True

        # Giả lập các thuộc tính của OpenAI CLIP để giữ nguyên code CIG
        self.visual = SimpleNamespace(
            input_resolution=image_encoder.config.image_size,
            output_dim=text_encoder.config.projection_dim
        )
        self.token_embedding = SimpleNamespace(
            embedding_dim=text_encoder.config.hidden_size
        )

    @property
    def dtype(self):
        return self.text_encoder.dtype

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        images = images.to(dtype=self.image_encoder.dtype)
        return self.image_encoder(images).image_embeds

    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        return self.text_encoder(text).text_embeds


def load_clip_model(clip_model_name: str, device: torch.device = device, download_root: str = "./"):
    """
    Tự động nhận diện và load OpenAI CLIP hoặc Hugging Face CLIP.
    """
    # Map các tên viết tắt của LinCIR (nếu có) sang repo Hugging Face chuẩn
    hf_model_map = {
        "large": "openai/clip-vit-large-patch14",
        "huge": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        "giga": "Geonmo/CLIP-Giga-config-fixed",
    }
    resolved_name = hf_model_map.get(clip_model_name, clip_model_name)

    # Nhánh 1: Nếu là model của OpenAI CLIP (ViT-B/32, ViT-L/14, ...)
    if resolved_name in clip.available_models() or os.path.isfile(resolved_name):
        print(f"Loading OpenAI CLIP model: {resolved_name}")
        clip_model, preprocess = clip.load(resolved_name, device=device, jit=False, download_root=download_root)
        clip_model.is_hf = False
        return clip_model, preprocess

    # Nhánh 2: Nếu là model trên Hugging Face (Geonmo/CLIP-Giga-config-fixed, ...)
    print(f"Loading HuggingFace CLIP model: {resolved_name}")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        resolved_name, torch_dtype=torch.float16, cache_dir=download_root
    ).to(device).eval()

    text_encoder = CLIPTextModelWithProjection.from_pretrained(
        resolved_name, torch_dtype=torch.float16, cache_dir=download_root
    ).to(device).eval()

    raw_processor = CLIPImageProcessor.from_pretrained(resolved_name, cache_dir=download_root)

    def hf_preprocess(img):
        return raw_processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)

    clip_model = HFCLIPWrapper(image_encoder, text_encoder).to(device).eval()
    return clip_model, hf_preprocess

@torch.no_grad()
def extract_image_features(dataset: Dataset, clip_model: CLIP, batch_size: Optional[int] = 32,
                           num_workers: Optional[int] = 10) -> Tuple[torch.Tensor, List[str]]:
    """
    Extracts image features from a dataset using a CLIP model.
    """
    # Create data loader
    loader = DataLoader(dataset=dataset, batch_size=batch_size,
                        num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)

    index_features = []
    index_names = []
    try:
        print(f"extracting image features {dataset.__class__.__name__} - {dataset.split}")
    except Exception as e:
        pass

    # Extract features
    for batch in tqdm(loader):
        images = batch.get('image')
        names = batch.get('image_name')
        if images is None:
            images = batch.get('reference_image')
        if names is None:
            names = batch.get('reference_name')

        images = images.to(device)
        with torch.no_grad():
            batch_features = clip_model.encode_image(images)
            index_features.append(batch_features.cpu())
            index_names.extend(names)

    index_features = torch.vstack(index_features)
    return index_features, index_names


def contrastive_loss(v1: torch.Tensor, v2: torch.Tensor, temperature: float) -> torch.Tensor:
    # Based on https://github.com/NVlabs/PALAVRA/blob/main/utils/nv.py
    v1 = F.normalize(v1, dim=1)
    v2 = F.normalize(v2, dim=1)

    numerator = torch.exp(torch.diag(torch.inner(v1, v2)) / temperature)
    numerator = torch.cat((numerator, numerator), 0)
    joint_vector = torch.cat((v1, v2), 0)
    pairs_product = torch.exp(torch.mm(joint_vector, joint_vector.t()) / temperature)
    denominator = torch.sum(pairs_product - pairs_product * torch.eye(joint_vector.shape[0]).to(device), 0)

    loss = -torch.mean(torch.log(numerator / denominator))

    return loss

@torch.no_grad()
def normalize_batch(batch):
    norm = torch.linalg.norm(batch, dim=1, keepdim=True)
    norm[norm == 0] = 1
    return batch / norm

@torch.no_grad()
# Function to compute slerp for a batch of tensors
def slerp_batch(v0, v1, t):
    v0 = normalize_batch(v0)
    v1 = normalize_batch(v1)

    dot = torch.sum(v0 * v1, dim=1, keepdim=True)
    dot = torch.clamp(dot, -1.0, 1.0)

    theta = torch.acos(dot)


    sin_theta = torch.sin(theta)
    weight0 = torch.sin((1 - t) * theta) / sin_theta
    weight1 = torch.sin(t * theta) / sin_theta


    weight0[sin_theta == 0] = 1 - t
    weight1[sin_theta == 0] = t

    interpolated = weight0 * v0 + weight1 * v1
    return interpolated
@torch.no_grad()
def extract_pseudo_tokens_with_phi(clip_model: CLIP, phi: Phi, dataset: Dataset, dataset_name="", l2_normalize: bool = False) -> Tuple[torch.Tensor, List[str]]:
    """
    Extracts pseudo tokens from a dataset using a CLIP model and a phi model
    """
    data_loader = DataLoader(dataset=dataset, batch_size=32, num_workers=10, pin_memory=False,
                             collate_fn=collate_fn)

    predicted_tokens = []
    predicted_tokens_gen = []
    names_list = []
    print(f"Extracting tokens using phi model (l2_normalize={l2_normalize})")
    for batch in tqdm(data_loader):
        images = batch.get('image')
        names = batch.get('image_name')
        if images is None:
            id = batch.get('pair_id')
            images_ref = batch.get('reference_image')
            images_gen = batch.get('generated_image')

        if names is None:
            names = batch.get('reference_name')
            names_t = batch.get('target_name')

        images_ref = images_ref.to(device)
        images_gen = images_gen.to(device)

        image_features_ref = clip_model.encode_image(images_ref).float()
        image_features_gen = clip_model.encode_image(images_gen).float()

        if l2_normalize:
            image_features_ref = F.normalize(image_features_ref, dim=-1)
            image_features_gen = F.normalize(image_features_gen, dim=-1)

        batch_predicted_tokens_ref = phi(image_features_ref)
        batch_predicted_tokens_gen = phi(image_features_gen)

        predicted_tokens.append(batch_predicted_tokens_ref.cpu())
        predicted_tokens_gen.append(batch_predicted_tokens_gen.cpu())

        if dataset_name == 'fashioniq':
            names_list.extend([x+y+str(z.item()) for x,y,z in zip(names,names_t,id)])
        else:
            names_list.extend(id)
    predicted_tokens = torch.vstack(predicted_tokens)
    predicted_tokens_gen = torch.vstack(predicted_tokens_gen)
    predicted_tokens = [predicted_tokens, predicted_tokens_gen]
    return predicted_tokens, names_list


class CustomTensorDataset(Dataset):
    """
    Custom Tensor Dataset which yields image_features and image_names
    """

    def __init__(self, images: torch.Tensor, names: torch.Tensor):
        self.images = images
        self.names = names

    def __getitem__(self, index) -> dict:
        return {'image': self.images[index],
                'image_name': self.names[index]
                }

    def __len__(self):
        return len(self.images)


def get_templates():
    """
    Return a list of templates
    Same templates as in PALAVRA: https://arxiv.org/abs/2204.01694
    """
    return [
        "This is a photo of a {}",
        "This photo contains a {}",
        "A photo of a {}",
        "This is an illustration of a {}",
        "This illustration contains a {}",
        "An illustrations of a {}",
        "This is a sketch of a {}",
        "This sketch contains a {}",
        "A sketch of a {}",
        "This is a diagram of a {}",
        "This diagram contains a {}",
        "A diagram of a {}",
        "A {}",
        "We see a {}",
        "{}",
        "We see a {} in this photo",
        "We see a {} in this image",
        "We see a {} in this illustration",
        "We see a {} photo",
        "We see a {} image",
        "We see a {} illustration",
        "{} photo",
        "{} image",
        "{} illustration",
    ]
