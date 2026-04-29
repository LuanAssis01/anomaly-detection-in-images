"""
Configurações gerais para o laboratório de detecção de anomalias
"""
import os

# Diretórios
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
TRAIN_DIR = os.path.join(DATA_DIR, 'train_images')
TEST_DIR = os.path.join(DATA_DIR, 'test_images')
TRAIN_MASKS_DIR = os.path.join(DATA_DIR, 'train_masks')
TRAIN_MASKS_VIS_DIR = os.path.join(DATA_DIR, 'train_masks_visualization')
SUPPLEMENTAL_MASKS_DIR = os.path.join(DATA_DIR, 'supplemental_masks')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
CHECKPOINTS_DIR = os.path.join(BASE_DIR, 'checkpoints')

# Fonte original das imagens reais (origem para todos os cenários)
SOURCE_TRAIN_DIR = os.path.join(DATA_DIR, 'train_images')
SOURCE_MASKS_DIR = os.path.join(DATA_DIR, 'train_masks')

# Diretórios por cenário
SCENARIOS_DIR = os.path.join(DATA_DIR, 'scenarios')
SCENARIOS = {
    # Apenas imagens reais, sem nenhuma transformação extra
    'no_augmentation': {
        'train_dir': os.path.join(SCENARIOS_DIR, 'no_augmentation', 'train_images'),
        'masks_dir': os.path.join(SCENARIOS_DIR, 'no_augmentation', 'train_masks'),
        'masks_vis_dir': os.path.join(SCENARIOS_DIR, 'no_augmentation', 'train_masks_visualization'),
    },
    # Imagens reais + augmentation para balancear as classes
    'no_synthetic': {
        'train_dir': os.path.join(SCENARIOS_DIR, 'no_synthetic', 'train_images'),
        'masks_dir': os.path.join(SCENARIOS_DIR, 'no_synthetic', 'train_masks'),
        'masks_vis_dir': os.path.join(SCENARIOS_DIR, 'no_synthetic', 'train_masks_visualization'),
    },
    # Imagens reais + augmentation + forjadas sintéticas geradas de autênticas
    'with_synthetic': {
        'train_dir': os.path.join(SCENARIOS_DIR, 'with_synthetic', 'train_images'),
        'masks_dir': os.path.join(SCENARIOS_DIR, 'with_synthetic', 'train_masks'),
        'masks_vis_dir': os.path.join(SCENARIOS_DIR, 'with_synthetic', 'train_masks_visualization'),
    },
}

# Parâmetros de treinamento
BATCH_SIZE = 6  # Reduzido: resolução 518x518 usa ~1.8x mais VRAM que 384x384
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4
DEVICE = 'cuda'  # 'cuda' ou 'cpu'
NUM_WORKERS = 4  # Workers paralelos para carregar dados (0 = sequencial, lento)

# Otimizações de performance (GPU)
USE_AMP = True                    # Mixed Precision (FP16) — ~2x mais rápido com Tensor Cores
GRADIENT_ACCUMULATION_STEPS = 2   # batch efetivo = 6*2 = 12 (compensa redução do batch size)
USE_COMPILE = False                # torch.compile() — funde operações para execução mais rápida

# Parâmetros das imagens
IMAGE_SIZE = 518  # 14×37 — resolução nativa do DINOv2 (patch_size=14), evita interpolação de embeddings
NUM_CLASSES = 2  # authentic vs forged

# Pesos das classes para penalizar Falsos Negativos (forged não detectado)
# classe 0 = authentic, classe 1 = forged (peso maior = penaliza mais FN)
CLASS_WEIGHTS = [1.0, 1.5]

# Configurações dos modelos
MODEL_CONFIGS = {
    'resnet50': {
        'name': 'ResNet-50',
        'model_name': 'resnet50',
        'num_classes': NUM_CLASSES
    },
    'resnet101': {
        'name': 'ResNet-101',
        'model_name': 'resnet101',
        'num_classes': NUM_CLASSES
    },
    'dinov2': {
        'name': 'DINOv2-Base',
        'model_name': 'facebook/dinov2-base',
        'num_classes': NUM_CLASSES
    },
    'dinov2_large': {
        'name': 'DINOv2-Large',
        'model_name': 'facebook/dinov2-large',
        'num_classes': NUM_CLASSES,
        'batch_size': 4,  # 307M params — menor batch para caber na VRAM com 518×518
    },
    'dinov3_small': {
        'name': 'DINOv3-Small',
        'model_name': 'facebook/dinov3-vits16-pretrain-lvd1689m',
        'num_classes': NUM_CLASSES,
    },
    'dinov3': {
        'name': 'DINOv3-Base',
        'model_name': 'facebook/dinov3-vitb16-pretrain-lvd1689m',
        'num_classes': NUM_CLASSES,
    },
    'dinov3_large': {
        'name': 'DINOv3-Large',
        'model_name': 'facebook/dinov3-vitl16-pretrain-lvd1689m',
        'num_classes': NUM_CLASSES,
        'batch_size': 4,  # ViT-L — menor batch para caber na VRAM
    },
}

# Proporções do split train/val/test (aplicadas sobre imagens originais)
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Threshold de decisão para classificação (< 0.5 favorece recall)
DECISION_THRESHOLD = 0.4

# Configurações de fine-tuning específicas por modelo (apenas ResNet-50 e DINOv2)
FINETUNE_CONFIGS = {
    'resnet101': {
        'phase1_epochs': 8,
        'phase1_lr': 6e-4,
        'phase2_epochs': 40,
        'phase2_backbone_lr': 5e-5,   # menor que resnet50 (7e-5): modelo maior, mais cuidado
        'phase2_classifier_lr': 3e-4,
        'warmup_epochs': 3,
        'weight_decay': 1e-3,
        'label_smoothing': 0.05,
        'early_stopping_patience': 12,
        'class_weights': [1.0, 1.5],
    },
    'resnet50': {
        # Fase 1: treinar só o classificador (backbone ImageNet congelado)
        'phase1_epochs': 8,
        'phase1_lr': 6e-4,            # otimizado: search encontrou ~6e-4
        # Fase 2: fine-tuning completo com lr diferenciado
        'phase2_epochs': 40,
        'phase2_backbone_lr': 7e-5,   # otimizado: era 2e-5, search encontrou 6-8e-5
        'phase2_classifier_lr': 3e-4, # otimizado: era 2e-4
        'warmup_epochs': 3,
        'weight_decay': 1e-3,         # otimizado: era 1e-4, search encontrou ~1e-3
        'label_smoothing': 0.05,
        'early_stopping_patience': 12,
        # Pesos por classe: search consistentemente preferiu [1.0, 1.5]
        'class_weights': [1.0, 1.5],
    },
    'dinov2': {
        'phase1_epochs': 12,
        'phase1_lr': 5e-4,
        'phase2_epochs': 35,
        'phase2_backbone_lr': 2e-6,   # otimizado: era 5e-6, 2-3e-6 mais estável
        'phase2_classifier_lr': 2e-4, # otimizado: era 5e-5, search encontrou ~1-4e-4
        'warmup_epochs': 4,
        'weight_decay': 2e-3,         # otimizado: era 0.05, search encontrou ~2e-3
        'label_smoothing': 0.05,
        'early_stopping_patience': 12,
        'class_weights': [1.0, 1.5],
    },
    'dinov2_large': {
        'phase1_epochs': 12,
        'phase1_lr': 3e-4,            # menor que base: modelo maior, mais cuidado na fase 1
        'phase2_epochs': 35,
        'phase2_backbone_lr': 1e-6,   # mais conservador que base (2e-6) — 307M params
        'phase2_classifier_lr': 2e-4,
        'warmup_epochs': 5,           # mais warmup por ser modelo maior
        'weight_decay': 2e-3,
        'label_smoothing': 0.05,
        'early_stopping_patience': 12,
        'class_weights': [1.0, 1.5],
    },
    # DINOv3 — patch_size=16, treinado no LVD-1689M
    # ATENÇÃO: backbone_lr muito alto causa colapso (recall ~0%), igual ao DINOv2
    'dinov3_small': {
        'phase1_epochs': 10,
        'phase1_lr': 6e-4,            # ViT-S é menor, suporta LR um pouco maior
        'phase2_epochs': 35,
        'phase2_backbone_lr': 3e-6,   # ViT-S pode ter backbone_lr ligeiramente maior que base
        'phase2_classifier_lr': 2e-4,
        'warmup_epochs': 3,
        'weight_decay': 2e-3,
        'label_smoothing': 0.05,
        'early_stopping_patience': 12,
        'class_weights': [1.0, 1.5],
    },
    'dinov3': {
        'phase1_epochs': 12,
        'phase1_lr': 5e-4,
        'phase2_epochs': 35,
        'phase2_backbone_lr': 2e-6,   # conservador — mesmo risco de colapso que DINOv2
        'phase2_classifier_lr': 2e-4,
        'warmup_epochs': 4,
        'weight_decay': 2e-3,
        'label_smoothing': 0.05,
        'early_stopping_patience': 12,
        'class_weights': [1.0, 1.5],
    },
    'dinov3_large': {
        'phase1_epochs': 12,
        'phase1_lr': 3e-4,            # ViT-L: LR conservador na fase 1
        'phase2_epochs': 35,
        'phase2_backbone_lr': 1e-6,   # muito conservador — modelo grande
        'phase2_classifier_lr': 2e-4,
        'warmup_epochs': 5,
        'weight_decay': 2e-3,
        'label_smoothing': 0.05,
        'early_stopping_patience': 12,
        'class_weights': [1.0, 1.5],
    },
}

# Configurações de geração de dados (balanceamento do dataset)
DATA_GENERATION = {
    'target_per_class': 3000,
    # Augmentation conservadora para autênticas (preserva features de autenticidade)
    'authentic_augmentation': {
        'horizontal_flip': True,
        'vertical_flip': False,
        'rotation_range': 5,
        'brightness_range': [0.95, 1.05],
        'zoom_range': 0.03,
    },
    # Augmentation agressiva para forjadas (todas as transformações válidas)
    'forged_augmentation': {
        'horizontal_flip': True,
        'vertical_flip': True,
        'rotation_range': 20,
        'brightness_range': [0.8, 1.2],
        'zoom_range': 0.15,
        'shear_range': 0.1,
    },
    # Proporções de técnicas de forjamento sintético
    'forgery_techniques': {
        'copy_move': 0.30,
        'splicing': 0.25,
        'inpainting': 0.20,
        'noise_injection': 0.15,
        'brightness_manipulation': 0.10,
    },
}

# Random seed para reprodutibilidade
RANDOM_SEED = 42
