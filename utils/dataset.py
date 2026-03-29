"""
Dataset customizado para detecção de falsificação em imagens
"""
import os
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Tuple, Optional, List


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
            for img_name in sorted(os.listdir(authentic_dir)):
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(authentic_dir, img_name))
                    self.labels.append(0)
                    self.mask_paths.append(None)  # Autênticas não têm máscaras

        # Imagens forjadas (classe 1)
        forged_dir = os.path.join(root_dir, 'forged')
        if os.path.exists(forged_dir):
            for img_name in sorted(os.listdir(forged_dir)):
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
    Retorna transformações para treino ou validação/teste.

    Modos:
        'train'      — augmentation conservadora aplicada igualmente a todas as imagens
                       (flip + rotação leve + colorjitter leve)
        'train_light' — alias de 'train', mantido por compatibilidade
        'val'/'test' — apenas resize + normalize (sem augmentation)

    Augmentation conservadora (mesma para ambas as classes):
      - RandomErasing, GaussianNoise e GaussianBlur foram removidos pois criam
        distribuições de treino/teste muito diferentes, levando ao colapso do modelo.
      - ColorJitter e rotação leve aumentam robustez sem destruir artefatos forenses.
    """
    if mode in ('train', 'train_light'):
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.03),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

def stratified_split(dataset: ForgeryDataset, val_ratio: float = 0.2,
                     seed: int = 42) -> tuple:
    """
    Split estratificado que mantém a proporção de classes igual
    em treino e validação.

    Args:
        dataset: ForgeryDataset com .labels
        val_ratio: Proporção para validação (default: 0.2)
        seed: Seed para reprodutibilidade

    Returns:
        (train_indices, val_indices)
    """
    from sklearn.model_selection import StratifiedShuffleSplit

    labels = np.array(dataset.labels)
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=val_ratio, random_state=seed
    )
    train_idx, val_idx = next(splitter.split(np.zeros(len(labels)), labels))

    # Verificar distribuição
    train_labels = labels[train_idx]
    val_labels = labels[val_idx]
    print(f"  Split estratificado:")
    print(f"    Treino:    {len(train_idx)} ({(train_labels == 0).sum()} auth, {(train_labels == 1).sum()} forged)")
    print(f"    Validação: {len(val_idx)} ({(val_labels == 0).sum()} auth, {(val_labels == 1).sum()} forged)")

    return train_idx.tolist(), val_idx.tolist()


def is_augmented_image(filepath: str) -> bool:
    """
    Verifica se uma imagem é augmentada/sintética (não original).

    Convenção do generate_data.py:
    - Originais: nomes numéricos (ex: 10.png, 10015.png)
    - Augmentadas autênticas: authentic_aug_NNNNN.png
    - Augmentadas forjadas: forged_aug_NNNNN.png
    - Sintéticas: synthetic_*_NNNNN.png
    """
    basename = os.path.basename(filepath)
    return any(basename.startswith(prefix) for prefix in [
        'authentic_aug_', 'forged_aug_', 'synthetic_'
    ])


class TransformSubset(Dataset):
    """
    Subset de um ForgeryDataset com transform próprio.
    Garante que train/val/test usem políticas de augmentation diferentes.
    """
    def __init__(self, base_dataset: 'ForgeryDataset', indices: List[int],
                 transform: transforms.Compose,
                 transform_light: Optional[transforms.Compose] = None):
        self.base_dataset = base_dataset
        self.indices = indices
        self.transform = transform
        self.transform_light = transform_light  # reservado, não usado

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Optional[torch.Tensor]]:
        real_idx = self.indices[idx]
        img_path = self.base_dataset.image_paths[real_idx]
        label = self.base_dataset.labels[real_idx]
        mask_path = self.base_dataset.mask_paths[real_idx]

        image = Image.open(img_path).convert('RGB')

        mask = None
        if mask_path is not None and os.path.exists(mask_path):
            try:
                mask = np.load(mask_path)
                mask = torch.from_numpy(mask).float()
                if mask.max() > 1:
                    mask = mask / 255.0
            except Exception:
                mask = None

        if self.transform:
            image = self.transform(image)

        return image, label, mask


def stratified_train_val_test_split(
    dataset: ForgeryDataset,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> tuple:
    """
    Split estratificado 3-way que previne vazamento de dados.

    - Imagens ORIGINAIS são divididas em train/val/test
    - Imagens AUGMENTADAS/SINTÉTICAS vão sempre para treino
    - Val e test contêm apenas imagens originais

    Args:
        dataset: ForgeryDataset instance
        val_ratio: Proporção de originais para validação
        test_ratio: Proporção de originais para teste
        seed: Random seed

    Returns:
        (train_indices, val_indices, test_indices)
    """
    from sklearn.model_selection import StratifiedShuffleSplit

    original_indices = []
    augmented_indices = []

    for i, path in enumerate(dataset.image_paths):
        if is_augmented_image(path):
            augmented_indices.append(i)
        else:
            original_indices.append(i)

    original_labels = np.array([dataset.labels[i] for i in original_indices])

    n_originals = len(original_indices)
    n_augmented = len(augmented_indices)

    print(f"  Imagens originais: {n_originals}")
    print(f"  Imagens augmentadas/sintéticas: {n_augmented}")

    # Split 1: originais → trainval vs test
    splitter1 = StratifiedShuffleSplit(
        n_splits=1, test_size=test_ratio, random_state=seed
    )
    trainval_rel, test_rel = next(splitter1.split(
        np.zeros(n_originals), original_labels
    ))

    # Split 2: trainval → train vs val
    trainval_labels = original_labels[trainval_rel]
    val_ratio_adjusted = val_ratio / (1 - test_ratio)
    splitter2 = StratifiedShuffleSplit(
        n_splits=1, test_size=val_ratio_adjusted, random_state=seed
    )
    train_rel_inner, val_rel_inner = next(splitter2.split(
        np.zeros(len(trainval_rel)), trainval_labels
    ))

    # Mapear de volta para índices do dataset
    train_original = [original_indices[trainval_rel[i]] for i in train_rel_inner]
    val_indices = [original_indices[trainval_rel[i]] for i in val_rel_inner]
    test_indices = [original_indices[i] for i in test_rel]

    # Augmentadas/sintéticas sempre vão para treino
    train_indices = train_original + augmented_indices

    # Distribuição
    train_labels = [dataset.labels[i] for i in train_indices]
    val_labels = [dataset.labels[i] for i in val_indices]
    test_labels = [dataset.labels[i] for i in test_indices]

    print(f"  Split 3-way (augmentadas → treino):")
    print(f"    Treino: {len(train_indices)} "
          f"({sum(1 for l in train_labels if l == 0)} auth, "
          f"{sum(1 for l in train_labels if l == 1)} forged)")
    print(f"    Val:    {len(val_indices)} "
          f"({sum(1 for l in val_labels if l == 0)} auth, "
          f"{sum(1 for l in val_labels if l == 1)} forged) [só originais]")
    print(f"    Teste:  {len(test_indices)} "
          f"({sum(1 for l in test_labels if l == 0)} auth, "
          f"{sum(1 for l in test_labels if l == 1)} forged) [só originais]")

    return train_indices, val_indices, test_indices


def save_split(split_dict: dict, path: str):
    """Salva índices do split em JSON para reutilização na avaliação."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(split_dict, f, indent=2)


def load_split(path: str) -> dict:
    """Carrega índices do split salvos durante o treino."""
    with open(path) as f:
        return json.load(f)


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
