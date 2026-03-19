"""
Geração e balanceamento de dataset para detecção de anomalias em imagens.

Utiliza Keras ImageDataGenerator para augmentation e técnicas de manipulação
(copy-move, splicing, inpainting) para gerar imagens forjadas sintéticas com
máscaras de anomalia correspondentes.

Uso:
    python src/generate_data.py --target 3000
    python src/generate_data.py --target 3000 --generate-forged-from-authentic
"""
import os
import sys
import argparse
import numpy as np
from PIL import Image, ImageFilter, ImageDraw
import random

# Adicionar raiz do projeto ao path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import (
    DATA_DIR, IMAGE_SIZE, RANDOM_SEED, SCENARIOS
)

# Keras imports
from keras_preprocessing.image import ImageDataGenerator

# Seed para reprodutibilidade
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# ============================================================================
# Configurações de geração
# ============================================================================

TARGET_PER_CLASS = 3000

# Augmentation CONSERVADORA para autênticas
# Objetivo: preservar features de autenticidade (padrões de sensor, compressão
# JPEG consistente, metadados visuais). Transformações mínimas que não
# destroem essas características.
AUTHENTIC_AUGMENTATION = {
    'horizontal_flip': True,
    'vertical_flip': False,           # Evitar: imagens invertidas perdem naturalidade
    'rotation_range': 5,              # Rotação mínima: preserva orientação natural
    'width_shift_range': 0.02,        # Shift mínimo: mantém composição original
    'height_shift_range': 0.02,
    'brightness_range': [0.95, 1.05], # Brilho quase imperceptível
    'zoom_range': 0.03,               # Zoom mínimo: preserva escala do sensor
    'channel_shift_range': 5.0,       # Shift de cor mínimo
    'fill_mode': 'reflect',           # Reflect mantém continuidade das bordas
}

# Augmentation AGRESSIVA para forjadas
# Objetivo: gerar variações diversas de manipulações existentes.
# Todas as transformações são válidas pois as features de forjamento
# (bordas inconsistentes, compressão dupla, etc.) são preservadas.
FORGED_AUGMENTATION = {
    'horizontal_flip': True,
    'vertical_flip': True,
    'rotation_range': 20,
    'width_shift_range': 0.1,
    'height_shift_range': 0.1,
    'brightness_range': [0.8, 1.2],
    'zoom_range': 0.15,
    'shear_range': 0.1,
    'channel_shift_range': 15.0,
    'fill_mode': 'reflect',
}

# ============================================================================
# Funções de manipulação para criar forjadas sintéticas
# ============================================================================


def copy_move_forgery(image: np.ndarray) -> tuple:
    """
    Copia uma região da imagem e cola em outra posição.
    Simula copy-move forgery — uma das manipulações mais comuns.

    Returns:
        (imagem_forjada, máscara_binária)
    """
    h, w = image.shape[:2]
    forged = image.copy()

    # Tamanho da região: entre 5% e 20% da imagem
    region_h = random.randint(int(h * 0.05), int(h * 0.20))
    region_w = random.randint(int(w * 0.05), int(w * 0.20))

    # Posição de origem (source)
    src_y = random.randint(0, h - region_h)
    src_x = random.randint(0, w - region_w)

    # Posição de destino (diferente da origem)
    dst_y = random.randint(0, h - region_h)
    dst_x = random.randint(0, w - region_w)

    # Garantir que destino é diferente da origem (overlap < 50%)
    attempts = 0
    while (abs(dst_y - src_y) < region_h // 2 and
           abs(dst_x - src_x) < region_w // 2) and attempts < 20:
        dst_y = random.randint(0, h - region_h)
        dst_x = random.randint(0, w - region_w)
        attempts += 1

    # Copiar região
    region = image[src_y:src_y + region_h, src_x:src_x + region_w].copy()

    # Aplicar leve transformação para dificultar detecção
    if random.random() > 0.5:
        # Leve blur para suavizar bordas
        pil_region = Image.fromarray(region)
        pil_region = pil_region.filter(ImageFilter.GaussianBlur(radius=0.5))
        region = np.array(pil_region)

    # Colar região
    forged[dst_y:dst_y + region_h, dst_x:dst_x + region_w] = region

    # Criar máscara binária (1 = região manipulada)
    mask = np.zeros((h, w), dtype=np.float32)
    mask[dst_y:dst_y + region_h, dst_x:dst_x + region_w] = 1.0

    return forged, mask


def splicing_forgery(image: np.ndarray, donor_image: np.ndarray) -> tuple:
    """
    Insere uma região de outra imagem (donor) na imagem atual.
    Simula image splicing — recortar e colar de fontes diferentes.

    Returns:
        (imagem_forjada, máscara_binária)
    """
    h, w = image.shape[:2]
    dh, dw = donor_image.shape[:2]
    forged = image.copy()

    # Tamanho da região do donor: entre 5% e 25% da imagem destino
    region_h = random.randint(int(h * 0.05), int(h * 0.25))
    region_w = random.randint(int(w * 0.05), int(w * 0.25))

    # Recortar do donor (redimensionar se necessário)
    donor_resized = np.array(
        Image.fromarray(donor_image).resize((max(region_w + 10, dw), max(region_h + 10, dh)))
    )
    dh2, dw2 = donor_resized.shape[:2]

    src_y = random.randint(0, max(0, dh2 - region_h))
    src_x = random.randint(0, max(0, dw2 - region_w))
    region = donor_resized[src_y:src_y + region_h, src_x:src_x + region_w]

    # Posição de destino
    dst_y = random.randint(0, h - region_h)
    dst_x = random.randint(0, w - region_w)

    # Blending suave nas bordas (feathering)
    mask_soft = _create_feathered_mask(region_h, region_w, feather=3)

    # Aplicar splice com blending
    for c in range(3):
        forged[dst_y:dst_y + region_h, dst_x:dst_x + region_w, c] = (
            region[:, :, c] * mask_soft +
            forged[dst_y:dst_y + region_h, dst_x:dst_x + region_w, c] * (1 - mask_soft)
        ).astype(np.uint8)

    # Máscara binária
    mask = np.zeros((h, w), dtype=np.float32)
    mask[dst_y:dst_y + region_h, dst_x:dst_x + region_w] = 1.0

    return forged, mask


def inpainting_forgery(image: np.ndarray) -> tuple:
    """
    Simula remoção de objeto via inpainting (blur forte em uma região).
    Útil para simular remoção/ocultação de elementos.

    Returns:
        (imagem_forjada, máscara_binária)
    """
    h, w = image.shape[:2]
    forged = image.copy()

    # Criar forma irregular para a região de inpainting
    mask = np.zeros((h, w), dtype=np.float32)
    pil_mask = Image.fromarray((mask * 255).astype(np.uint8))
    draw = ImageDraw.Draw(pil_mask)

    # Gerar polígono irregular
    center_y = random.randint(int(h * 0.2), int(h * 0.8))
    center_x = random.randint(int(w * 0.2), int(w * 0.8))
    num_points = random.randint(5, 10)
    radius = random.randint(int(min(h, w) * 0.03), int(min(h, w) * 0.12))

    points = []
    for i in range(num_points):
        angle = 2 * np.pi * i / num_points
        r = radius * (0.7 + 0.6 * random.random())
        px = int(center_x + r * np.cos(angle))
        py = int(center_y + r * np.sin(angle))
        px = max(0, min(w - 1, px))
        py = max(0, min(h - 1, py))
        points.append((px, py))

    draw.polygon(points, fill=255)
    mask = np.array(pil_mask).astype(np.float32) / 255.0

    # Aplicar blur forte na região mascarada (simula inpainting)
    pil_image = Image.fromarray(forged)
    blurred = pil_image.filter(ImageFilter.GaussianBlur(radius=8))
    blurred_arr = np.array(blurred)

    # Combinar: blur onde mask=1, original onde mask=0
    mask_3d = np.expand_dims(mask, axis=2)
    forged = (blurred_arr * mask_3d + forged * (1 - mask_3d)).astype(np.uint8)

    return forged, mask


def noise_injection_forgery(image: np.ndarray) -> tuple:
    """
    Injeta ruído em uma região específica da imagem.
    Simula artefatos de compressão ou edição localizada.

    Returns:
        (imagem_forjada, máscara_binária)
    """
    h, w = image.shape[:2]
    forged = image.copy()

    # Região retangular com ruído
    region_h = random.randint(int(h * 0.05), int(h * 0.20))
    region_w = random.randint(int(w * 0.05), int(w * 0.20))
    y = random.randint(0, h - region_h)
    x = random.randint(0, w - region_w)

    # Gerar ruído gaussiano
    noise_std = random.uniform(10, 30)
    noise = np.random.normal(0, noise_std, (region_h, region_w, 3))
    forged[y:y + region_h, x:x + region_w] = np.clip(
        forged[y:y + region_h, x:x + region_w].astype(np.float32) + noise,
        0, 255
    ).astype(np.uint8)

    # Máscara binária
    mask = np.zeros((h, w), dtype=np.float32)
    mask[y:y + region_h, x:x + region_w] = 1.0

    return forged, mask


def brightness_manipulation_forgery(image: np.ndarray) -> tuple:
    """
    Altera brilho/contraste de uma região específica.
    Simula edição localizada (Photoshop dodge/burn).

    Returns:
        (imagem_forjada, máscara_binária)
    """
    h, w = image.shape[:2]
    forged = image.copy().astype(np.float32)

    # Região circular
    center_y = random.randint(int(h * 0.2), int(h * 0.8))
    center_x = random.randint(int(w * 0.2), int(w * 0.8))
    radius = random.randint(int(min(h, w) * 0.05), int(min(h, w) * 0.15))

    # Criar máscara circular suave
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((Y - center_y) ** 2 + (X - center_x) ** 2)
    mask = np.clip(1.0 - dist / radius, 0, 1).astype(np.float32)

    # Alterar brilho na região
    factor = random.choice([random.uniform(0.6, 0.8), random.uniform(1.2, 1.5)])
    mask_3d = np.expand_dims(mask, axis=2)
    forged = forged * (1 + mask_3d * (factor - 1))
    forged = np.clip(forged, 0, 255).astype(np.uint8)

    # Binarizar máscara para o output (threshold > 0.5)
    mask_binary = (mask > 0.3).astype(np.float32)

    return forged, mask_binary


def _normalize_mask_shape(mask: np.ndarray, target_size: tuple = None) -> np.ndarray:
    """
    Garante que a máscara seja 2D (H, W) float32 em [0, 1].
    Aceita qualquer shape com dimensões extras (1, H, W), (H, W, 1), (1, 1, W), etc.
    """
    # Remover todas as dimensões de tamanho 1 até restar 2D
    mask = np.squeeze(mask)

    # Se ficou 1D (caso raro de máscara malformada), tentar reshape
    if mask.ndim == 1:
        side = int(np.sqrt(mask.size))
        if side * side == mask.size:
            mask = mask.reshape(side, side)
        else:
            # Fallback: máscara vazia no tamanho alvo
            h, w = target_size[1], target_size[0] if target_size else (mask.size, 1)
            mask = np.zeros((h, w), dtype=np.float32)

    # Se ainda tem mais de 2 dimensões (ex: H, W, C), pegar apenas o primeiro canal
    while mask.ndim > 2:
        mask = mask[..., 0] if mask.shape[-1] <= mask.shape[0] else mask[0]

    mask = mask.astype(np.float32)

    # Redimensionar para target size se necessário
    if target_size is not None:
        th, tw = target_size[1], target_size[0]
        if mask.shape[0] != th or mask.shape[1] != tw:
            mask_pil = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
            mask_pil = mask_pil.resize((tw, th), Image.NEAREST)
            mask = np.array(mask_pil).astype(np.float32) / 255.0

    return mask


def _create_feathered_mask(height: int, width: int, feather: int = 3) -> np.ndarray:
    """Cria máscara com bordas suavizadas (feathering) para blending."""
    mask = np.ones((height, width), dtype=np.float32)
    for i in range(feather):
        alpha = (i + 1) / (feather + 1)
        mask[i, :] *= alpha
        mask[-(i + 1), :] *= alpha
        mask[:, i] *= alpha
        mask[:, -(i + 1)] *= alpha
    return mask


# ============================================================================
# Funções de geração de dataset
# ============================================================================


def load_images_from_dir(directory: str, target_size: tuple = None) -> list:
    """Carrega todas as imagens de um diretório."""
    images = []
    if not os.path.exists(directory):
        return images

    for fname in sorted(os.listdir(directory)):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(directory, fname)
            img = Image.open(img_path).convert('RGB')
            if target_size:
                img = img.resize(target_size, Image.LANCZOS)
            images.append((fname, np.array(img)))

    return images


def augment_with_keras(images: list, augmentation_config: dict,
                       num_to_generate: int, prefix: str) -> list:
    """
    Gera imagens augmentadas usando Keras ImageDataGenerator.

    Args:
        images: Lista de (nome, np_array) das imagens originais
        augmentation_config: Config do ImageDataGenerator
        num_to_generate: Quantas imagens gerar
        prefix: Prefixo para nomes dos arquivos gerados

    Returns:
        Lista de (nome, np_array) das imagens geradas
    """
    if not images or num_to_generate <= 0:
        return []

    datagen = ImageDataGenerator(**augmentation_config)

    generated = []
    count = 0
    img_idx = 0

    while count < num_to_generate:
        # Ciclar pelas imagens originais
        _, img_array = images[img_idx % len(images)]
        img_idx += 1

        # Keras espera shape (1, H, W, C)
        img_batch = np.expand_dims(img_array, axis=0)

        # Gerar 1 imagem augmentada
        for batch in datagen.flow(img_batch, batch_size=1):
            aug_img = batch[0].astype(np.uint8)
            name = f"{prefix}_aug_{count:05d}.png"
            generated.append((name, aug_img))
            count += 1
            break

        if count % 500 == 0 and count > 0:
            print(f"  Geradas {count}/{num_to_generate} imagens {prefix}")

    return generated


def augment_forged_with_mask(images: list, masks: list,
                             augmentation_config: dict,
                             num_to_generate: int) -> tuple:
    """
    Gera imagens forjadas augmentadas com Keras, aplicando as MESMAS
    transformações geométricas na máscara correspondente.

    Para manter a consistência imagem-máscara, as transformações de cor
    (brightness, channel_shift) são aplicadas só na imagem, enquanto
    transformações geométricas (flip, rotation, shift, zoom) são aplicadas
    em ambas.

    Returns:
        (lista_imagens, lista_mascaras)
    """
    if not images or num_to_generate <= 0:
        return [], []

    # Separar configs geométricas e de cor
    geometric_config = {
        'horizontal_flip': augmentation_config.get('horizontal_flip', False),
        'vertical_flip': augmentation_config.get('vertical_flip', False),
        'rotation_range': augmentation_config.get('rotation_range', 0),
        'width_shift_range': augmentation_config.get('width_shift_range', 0),
        'height_shift_range': augmentation_config.get('height_shift_range', 0),
        'zoom_range': augmentation_config.get('zoom_range', 0),
        'shear_range': augmentation_config.get('shear_range', 0),
        'fill_mode': augmentation_config.get('fill_mode', 'reflect'),
    }

    datagen_img = ImageDataGenerator(**augmentation_config)
    datagen_mask = ImageDataGenerator(**geometric_config)

    generated_imgs = []
    generated_masks = []
    count = 0
    idx = 0

    while count < num_to_generate:
        i = idx % len(images)
        _, img_array = images[i]
        mask_array = masks[i]
        idx += 1

        # Usar mesma seed para manter transformações geométricas sincronizadas
        seed = np.random.randint(0, 2**31)

        img_batch = np.expand_dims(img_array, axis=0)  # (1, H, W, 3)

        # Garantir que máscara é 2D e tem mesmo tamanho da imagem
        h, w = img_array.shape[:2]
        mask_array = _normalize_mask_shape(mask_array, target_size=(w, h))

        # Máscara precisa ser (1, H, W, 1) para Keras
        mask_3ch = mask_array[np.newaxis, :, :, np.newaxis]  # (1, H, W, 1)

        for aug_img in datagen_img.flow(img_batch, batch_size=1, seed=seed):
            aug_img = aug_img[0].astype(np.uint8)
            break

        for aug_mask in datagen_mask.flow(mask_3ch, batch_size=1, seed=seed):
            aug_mask = aug_mask[0, :, :, 0]
            # Binarizar máscara após transformação (pode ter valores intermediários)
            aug_mask = (aug_mask > 0.5).astype(np.float32)
            break

        name = f"forged_aug_{count:05d}.png"
        generated_imgs.append((name, aug_img))
        generated_masks.append((name, aug_mask))
        count += 1

        if count % 500 == 0 and count > 0:
            print(f"  Geradas {count}/{num_to_generate} imagens forjadas com máscara")

    return generated_imgs, generated_masks


def generate_synthetic_forged(authentic_images: list,
                              num_to_generate: int) -> tuple:
    """
    Cria imagens forjadas sintéticas a partir de imagens autênticas,
    aplicando diferentes tipos de manipulação. Cada imagem recebe uma
    máscara binária indicando a região manipulada.

    Técnicas aplicadas:
        - Copy-move (30%): copia e cola região dentro da mesma imagem
        - Splicing (25%): insere região de outra imagem
        - Inpainting (20%): simula remoção de objeto via blur
        - Noise injection (15%): injeta ruído localizado
        - Brightness manipulation (10%): edição localizada de brilho

    Returns:
        (lista_imagens, lista_mascaras)
    """
    if not authentic_images or num_to_generate <= 0:
        return [], []

    forgery_functions = [
        ('copy_move', 0.30, copy_move_forgery),
        ('splicing', 0.25, None),  # Tratado separadamente (precisa de donor)
        ('inpainting', 0.20, inpainting_forgery),
        ('noise', 0.15, noise_injection_forgery),
        ('brightness', 0.10, brightness_manipulation_forgery),
    ]

    generated_imgs = []
    generated_masks = []

    for i in range(num_to_generate):
        # Selecionar tipo de manipulação por peso
        r = random.random()
        cumulative = 0
        selected_type = 'copy_move'
        selected_func = copy_move_forgery

        for ftype, prob, func in forgery_functions:
            cumulative += prob
            if r <= cumulative:
                selected_type = ftype
                selected_func = func
                break

        # Selecionar imagem base
        _, base_img = random.choice(authentic_images)

        if selected_type == 'splicing':
            # Splicing precisa de uma imagem donor diferente
            _, donor_img = random.choice(authentic_images)
            forged_img, mask = splicing_forgery(base_img, donor_img)
        else:
            forged_img, mask = selected_func(base_img)

        name = f"synthetic_{selected_type}_{i:05d}.png"
        generated_imgs.append((name, forged_img))
        generated_masks.append((name, mask))

        if (i + 1) % 500 == 0:
            print(f"  Geradas {i + 1}/{num_to_generate} forjadas sintéticas")

    return generated_imgs, generated_masks


def save_images(images: list, output_dir: str):
    """Salva lista de (nome, np_array) como arquivos de imagem."""
    os.makedirs(output_dir, exist_ok=True)
    for name, img_array in images:
        img = Image.fromarray(img_array)
        img.save(os.path.join(output_dir, name))


def save_masks(masks: list, output_dir: str):
    """Salva lista de (nome, np_array) como arquivos .npy e .png."""
    os.makedirs(output_dir, exist_ok=True)
    for name, mask_array in masks:
        mask_name = name.replace('.png', '.npy').replace('.jpg', '.npy')
        np.save(os.path.join(output_dir, mask_name), mask_array)
        # Salvar máscara como PNG (branco = região manipulada)
        mask_png_name = mask_name.replace('.npy', '_mask.png')
        mask_img = Image.fromarray((mask_array * 255).astype(np.uint8), mode='L')
        mask_img.save(os.path.join(output_dir, mask_png_name))


def save_mask_overlays(images: list, masks: list, output_dir: str,
                       alpha: float = 0.4):
    """
    Salva visualizações overlay: imagem forjada com máscara vermelha sobreposta.
    Permite identificar visualmente a região manipulada.

    Args:
        images: Lista de (nome, np_array) das imagens forjadas
        masks: Lista de (nome, np_array) das máscaras correspondentes
        output_dir: Diretório de saída para as visualizações
        alpha: Opacidade do overlay vermelho (0=transparente, 1=opaco)
    """
    os.makedirs(output_dir, exist_ok=True)
    for (img_name, img_array), (_, mask_array) in zip(images, masks):
        h, w = img_array.shape[:2]
        mask_resized = _normalize_mask_shape(mask_array, target_size=(w, h))

        # Criar overlay vermelho na região manipulada
        overlay = img_array.copy().astype(np.float32)
        mask_3d = np.expand_dims(mask_resized, axis=2)

        # Vermelho semi-transparente onde mask=1
        red = np.array([255.0, 0.0, 0.0])
        overlay = overlay * (1 - mask_3d * alpha) + red * mask_3d * alpha
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        # Desenhar contorno verde ao redor da região manipulada
        mask_uint8 = (mask_resized * 255).astype(np.uint8)
        mask_pil = Image.fromarray(mask_uint8, mode='L')
        # Detectar bordas via dilatação - erosão
        dilated = np.array(mask_pil.filter(ImageFilter.MaxFilter(5)))
        eroded = np.array(mask_pil.filter(ImageFilter.MinFilter(5)))
        contour = ((dilated > 127) & ~(eroded > 127))

        overlay[contour] = [0, 255, 0]  # Contorno verde

        vis_name = img_name.replace('.png', '_overlay.png').replace('.jpg', '_overlay.png')
        Image.fromarray(overlay).save(os.path.join(output_dir, vis_name))


# ============================================================================
# Pipeline principal
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description='Gerar e balancear dataset de imagens autênticas e forjadas'
    )
    parser.add_argument(
        '--target', type=int, default=TARGET_PER_CLASS,
        help=f'Número alvo de imagens por classe (default: {TARGET_PER_CLASS})'
    )
    parser.add_argument(
        '--scenario', type=str, default='no_synthetic',
        choices=['no_synthetic', 'with_synthetic'],
        help=(
            'Cenário de geração:\n'
            '  no_synthetic   — apenas augmentation de imagens reais (default)\n'
            '  with_synthetic — também gera forjadas sintéticas a partir de autênticas'
        )
    )
    parser.add_argument(
        '--image-size', type=int, default=IMAGE_SIZE,
        help=f'Tamanho das imagens geradas (default: {IMAGE_SIZE})'
    )
    parser.add_argument(
        '--synthetic-ratio', type=float, default=0.5,
        help='Proporção de forjadas sintéticas vs augmentadas (default: 0.5, só para with_synthetic)'
    )
    args = parser.parse_args()

    target = args.target
    size = (args.image_size, args.image_size)
    scenario = args.scenario
    # with_synthetic ativa geração de forjadas sintéticas automaticamente
    generate_synthetic = (scenario == 'with_synthetic')

    print("=" * 60)
    print("GERAÇÃO E BALANCEAMENTO DE DATASET")
    print(f"Cenário:  {scenario}")
    print(f"Alvo:     {target} imagens por classe")
    print(f"Tamanho:  {size[0]}x{size[1]}")
    print("=" * 60)

    # Diretórios baseados no cenário
    scenario_cfg = SCENARIOS[scenario]
    train_dir = scenario_cfg['train_dir']
    masks_dir = scenario_cfg['masks_dir']
    masks_vis_dir = scenario_cfg['masks_vis_dir']

    authentic_dir = os.path.join(train_dir, 'authentic')
    forged_dir = os.path.join(train_dir, 'forged')

    print(f"\nDiretório de saída: {train_dir}")

    os.makedirs(authentic_dir, exist_ok=True)
    os.makedirs(forged_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(masks_vis_dir, exist_ok=True)

    # ----------------------------------------------------------------
    # 1. Carregar imagens existentes
    # ----------------------------------------------------------------
    print("\n[1/5] Carregando imagens existentes...")
    authentic_images = load_images_from_dir(authentic_dir, target_size=size)
    forged_images = load_images_from_dir(forged_dir, target_size=size)

    # Carregar máscaras existentes para forjadas (redimensionar para target size)
    existing_masks = {}
    if os.path.exists(masks_dir):
        for fname in os.listdir(masks_dir):
            if fname.endswith('.npy'):
                mask = np.load(os.path.join(masks_dir, fname))
                # Normalizar para float [0, 1]
                if mask.max() > 1:
                    mask = mask / 255.0
                # Garantir shape 2D (H, W)
                mask = _normalize_mask_shape(mask, size)
                existing_masks[fname] = mask

    n_authentic = len(authentic_images)
    n_forged = len(forged_images)
    print(f"  Autênticas encontradas: {n_authentic}")
    print(f"  Forjadas encontradas: {n_forged}")

    # Gerar visualizações para forjadas existentes que já possuem máscara
    existing_with_mask = []
    existing_mask_list = []
    for name, img_array in forged_images:
        mask_name = name.replace('.png', '.npy').replace('.jpg', '.npy').replace('.jpeg', '.npy')
        if mask_name in existing_masks:
            existing_with_mask.append((name, img_array))
            existing_mask_list.append((name, existing_masks[mask_name]))
    if existing_with_mask:
        print(f"  Gerando visualizações para {len(existing_with_mask)} forjadas existentes...")
        save_mask_overlays(existing_with_mask, existing_mask_list, masks_vis_dir)
        # Salvar máscaras existentes como PNG também
        for name, mask in existing_mask_list:
            mask_png_name = name.replace('.png', '_mask.png').replace('.jpg', '_mask.png').replace('.jpeg', '_mask.png')
            mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
            mask_img.save(os.path.join(masks_dir, mask_png_name))

    if n_authentic == 0 and n_forged == 0:
        print("\n[ERRO] Nenhuma imagem encontrada!")
        print(f"  Coloque imagens autênticas em: {authentic_dir}")
        print(f"  Coloque imagens forjadas em: {forged_dir}")
        print("  Formatos aceitos: .png, .jpg, .jpeg")
        sys.exit(1)

    # ----------------------------------------------------------------
    # 2. Gerar imagens autênticas (augmentation conservadora)
    # ----------------------------------------------------------------
    need_authentic = max(0, target - n_authentic)
    print(f"\n[2/5] Gerando {need_authentic} imagens autênticas augmentadas...")

    if need_authentic > 0 and n_authentic > 0:
        aug_authentic = augment_with_keras(
            authentic_images, AUTHENTIC_AUGMENTATION,
            need_authentic, prefix='authentic'
        )
        save_images(aug_authentic, authentic_dir)
        print(f"  Salvas {len(aug_authentic)} imagens autênticas em {authentic_dir}")
    elif n_authentic == 0:
        print("  [AVISO] Sem imagens autênticas para augmentar.")
        print("  Use --scenario with_synthetic para criar forjadas a partir delas.")

    # ----------------------------------------------------------------
    # 3. Gerar imagens forjadas sintéticas (a partir de autênticas)
    # ----------------------------------------------------------------
    synthetic_imgs = []
    synthetic_masks = []
    need_forged = max(0, target - n_forged)

    if generate_synthetic and n_authentic > 0:
        n_synthetic = int(need_forged * args.synthetic_ratio)
        print(f"\n[3/5] Gerando {n_synthetic} forjadas sintéticas a partir de autênticas...")

        synthetic_imgs, synthetic_masks = generate_synthetic_forged(
            authentic_images, n_synthetic
        )
        save_images(synthetic_imgs, forged_dir)
        save_masks(synthetic_masks, masks_dir)
        save_mask_overlays(synthetic_imgs, synthetic_masks, masks_vis_dir)
        print(f"  Salvas {len(synthetic_imgs)} forjadas sintéticas com máscaras e visualizações")
        need_forged -= len(synthetic_imgs)
    else:
        print("\n[3/5] Pulando geração de forjadas sintéticas (cenário: no_synthetic)")
        if generate_synthetic and n_authentic == 0:
            print("  [AVISO] Sem imagens autênticas para criar forjadas sintéticas")

    # ----------------------------------------------------------------
    # 4. Augmentar imagens forjadas existentes (com máscaras sincronizadas)
    # ----------------------------------------------------------------
    print(f"\n[4/5] Gerando {max(0, need_forged)} forjadas augmentadas...")

    # Reunir todas as forjadas (originais + sintéticas) com suas máscaras
    all_forged = forged_images + synthetic_imgs
    all_masks = []
    for name, _ in forged_images:
        mask_name = name.replace('.png', '.npy').replace('.jpg', '.npy').replace('.jpeg', '.npy')
        if mask_name in existing_masks:
            all_masks.append(existing_masks[mask_name])
        else:
            # Sem máscara -> criar máscara vazia (será toda preta)
            all_masks.append(np.zeros((size[1], size[0]), dtype=np.float32))
    for name, mask in synthetic_masks:
        all_masks.append(mask)

    if need_forged > 0 and len(all_forged) > 0:
        # Preparar pares (imagem, máscara)
        forged_with_masks = list(zip(
            [(n, img) for n, img in all_forged],
            all_masks
        ))
        imgs_list = [x[0] for x in forged_with_masks]
        masks_list = [x[1] for x in forged_with_masks]

        aug_forged_imgs, aug_forged_masks = augment_forged_with_mask(
            imgs_list, masks_list, FORGED_AUGMENTATION, need_forged
        )
        save_images(aug_forged_imgs, forged_dir)
        save_masks(aug_forged_masks, masks_dir)
        save_mask_overlays(aug_forged_imgs, aug_forged_masks, masks_vis_dir)
        print(f"  Salvas {len(aug_forged_imgs)} forjadas augmentadas com máscaras e visualizações")
    elif n_forged == 0 and len(synthetic_imgs) == 0:
        print("  [AVISO] Sem imagens forjadas para augmentar.")

    # ----------------------------------------------------------------
    # 5. Relatório final
    # ----------------------------------------------------------------
    print(f"\n[5/5] Relatório final")
    print("=" * 60)

    final_authentic = len([
        f for f in os.listdir(authentic_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    final_forged = len([
        f for f in os.listdir(forged_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    final_masks = len([
        f for f in os.listdir(masks_dir) if f.endswith('.npy')
    ])
    final_masks_png = len([
        f for f in os.listdir(masks_dir) if f.endswith('_mask.png')
    ])
    final_overlays = len([
        f for f in os.listdir(masks_vis_dir) if f.endswith('_overlay.png')
    ]) if os.path.exists(masks_vis_dir) else 0

    print(f"  Imagens autênticas: {final_authentic}")
    print(f"  Imagens forjadas:   {final_forged}")
    print(f"  Máscaras (.npy):    {final_masks}")
    print(f"  Máscaras (.png):    {final_masks_png}")
    print(f"  Overlays gerados:   {final_overlays}")
    print(f"  Total de imagens:   {final_authentic + final_forged}")
    print(f"  Balanceamento:      {'OK' if final_authentic == final_forged else 'DESBALANCEADO'}")
    print("=" * 60)

    if final_authentic != target or final_forged != target:
        print(f"\n  [AVISO] Alvo era {target} por classe.")
        if final_authentic < target:
            print(f"  Faltam {target - final_authentic} autênticas (adicione mais imagens originais)")
        if final_forged < target:
            print(f"  Faltam {target - final_forged} forjadas (adicione mais ou use --generate-forged-from-authentic)")


if __name__ == '__main__':
    main()
