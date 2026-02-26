"""
Dataset customizado para detecção de falsificação em imagens
"""
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Tuple, Optional


class GaussianNoise:
    """Adiciona ruído gaussiano ao tensor (simula artefatos de compressão JPEG)"""
    def __init__(self, p=0.2, std=0.02):
        self.p = p
        self.std = std

    def __call__(self, tensor):
        if torch.rand(1).item() < self.p:
            noise = torch.randn_like(tensor) * self.std
            return torch.clamp(tensor + noise, 0.0, 1.0)
        return tensor

class ForgeryDataset(Dataset):
    """
    Dataset para classificação e segmentação de imagens forjadas
    """
    def __init__(
        self, 
        root_dir: str, 
        masks_dir: Optional[str] = None,
        transform: Optional[transforms.Compose] = None,
        mode: str = 'train'
    ):
        """
        Args:
            root_dir: Diretório raiz com subpastas 'authentic' e 'forged'
            masks_dir: Diretório com máscaras de segmentação (.npy)
            transform: Transformações a serem aplicadas
            mode: 'train' ou 'test'
        """
        self.root_dir = root_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.mode = mode
        
        # Carregar paths das imagens
        self.image_paths = []
        self.labels = []
        self.mask_paths = []
        
        # Imagens autênticas (classe 0)
        authentic_dir = os.path.join(root_dir, 'authentic')
        if os.path.exists(authentic_dir):
            for img_name in os.listdir(authentic_dir):
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(authentic_dir, img_name))
                    self.labels.append(0)
                    self.mask_paths.append(None)  # Autênticas não têm máscaras
        
        # Imagens forjadas (classe 1)
        forged_dir = os.path.join(root_dir, 'forged')
        if os.path.exists(forged_dir):
            for img_name in os.listdir(forged_dir):
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(forged_dir, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(1)
                    
                    # Verificar se existe máscara correspondente
                    mask_name = img_name.replace('.png', '.npy').replace('.jpg', '.npy').replace('.jpeg', '.npy')
                    if masks_dir:
                        mask_path = os.path.join(masks_dir, mask_name)
                        self.mask_paths.append(mask_path if os.path.exists(mask_path) else None)
                    else:
                        self.mask_paths.append(None)
        
        print(f"Dataset carregado: {len(self.image_paths)} imagens")
        print(f"  - Autênticas: {self.labels.count(0)}")
        print(f"  - Forjadas: {self.labels.count(1)}")
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Optional[torch.Tensor]]:
        """
        Retorna: (imagem, label, máscara)
        """
        # Carregar imagem
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Carregar máscara se existir
        mask = None
        if self.mask_paths[idx] is not None and os.path.exists(self.mask_paths[idx]):
            try:
                mask = np.load(self.mask_paths[idx])
                mask = torch.from_numpy(mask).float()
                # Normalizar máscara para [0, 1]
                if mask.max() > 1:
                    mask = mask / 255.0
            except Exception as e:
                print(f"Erro ao carregar máscara {self.mask_paths[idx]}: {e}")
                mask = None
        
        # Aplicar transformações
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        
        return image, label, mask

def get_transforms(image_size: int = 224, mode: str = 'train') -> transforms.Compose:
    """
    Retorna transformações apropriadas para treino ou validação
    """
    if mode == 'train':
        from configs.config import AUGMENTATION
        transform_list = [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=AUGMENTATION.get('horizontal_flip', 0.5)),
            transforms.RandomVerticalFlip(p=AUGMENTATION.get('vertical_flip', 0.3)),
            transforms.RandomRotation(AUGMENTATION.get('rotation_degrees', 15)),
            transforms.ColorJitter(
                brightness=AUGMENTATION.get('brightness', 0.2),
                contrast=AUGMENTATION.get('contrast', 0.2),
                saturation=AUGMENTATION.get('saturation', 0.0),
                hue=AUGMENTATION.get('hue', 0.0),
            ),
        ]
        # Gaussian Blur — simula degradação de qualidade (aplicado antes de ToTensor)
        if AUGMENTATION.get('gauss_blur', 0) > 0:
            transform_list.append(
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))],
                    p=AUGMENTATION['gauss_blur']
                )
            )
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
        ])
        # Gaussian Noise — simula artefatos de compressão (aplicado após ToTensor)
        gauss_noise_std = AUGMENTATION.get('gauss_noise_std', 0)
        if gauss_noise_std > 0:
            transform_list.append(GaussianNoise(p=0.2, std=gauss_noise_std))
        # Random Erasing — aplicado após ToTensor/Normalize
        if AUGMENTATION.get('random_erasing', 0) > 0:
            transform_list.append(
                transforms.RandomErasing(p=AUGMENTATION['random_erasing'], scale=(0.02, 0.15))
            )
        return transforms.Compose(transform_list)
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

def custom_collate_fn(batch):
    """
    Collate function customizado para lidar com máscaras que podem ser None
    
    Args:
        batch: Lista de tuplas (image, label, mask)
    
    Returns:
        images: Tensor [B, C, H, W]
        labels: Tensor [B]
        masks: Lista de tensors ou None
    """
    images = []
    labels = []
    masks = []
    
    for image, label, mask in batch:
        images.append(image)
        labels.append(label)
        masks.append(mask)  # Pode ser None
    
    # Stack images e labels
    images = torch.stack(images, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)
    
    # Manter masks como lista (pode conter None)
    return images, labels, masks
