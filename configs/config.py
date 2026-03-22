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
BATCH_SIZE = 8  # Reduzido de 16: resolução 384x384 usa ~3x mais VRAM que 224x224
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4
DEVICE = 'cuda'  # 'cuda' ou 'cpu'
NUM_WORKERS = 4  # Workers paralelos para carregar dados (0 = sequencial, lento)

# Otimizações de performance (GPU)
USE_AMP = True                    # Mixed Precision (FP16) — ~2x mais rápido com Tensor Cores
GRADIENT_ACCUMULATION_STEPS = 2   # batch efetivo = 8*2 = 16 (compensa redução do batch size)
USE_COMPILE = False                # torch.compile() — funde operações para execução mais rápida

# Parâmetros das imagens
IMAGE_SIZE = 384  # Aumentado: falsificações sutis (~2% da área) precisam de mais resolução
NUM_CLASSES = 2  # authentic vs forged

# Pesos das classes para penalizar Falsos Negativos (forged não detectado)
# classe 0 = authentic, classe 1 = forged (peso maior = penaliza mais FN)
CLASS_WEIGHTS = [1.0, 1.5]

# Configurações dos modelos (apenas ResNet-50 e DINOv2)
MODEL_CONFIGS = {
    'resnet50': {
        'name': 'ResNet-50',
        'model_name': 'resnet50',
        'num_classes': NUM_CLASSES
    },
    'dinov2': {
        'name': 'DINOv2',
        'model_name': 'facebook/dinov2-base',
        'num_classes': NUM_CLASSES
    }
}

# Configurações de data augmentation
AUGMENTATION = {
    'horizontal_flip': 0.5,
    'vertical_flip': 0.3,
    'rotation_degrees': 15,
    'brightness': 0.15,
    'contrast': 0.15,
    'saturation': 0.1,
    'hue': 0.08,
    'random_erasing': 0.3,   # máscara aleatória — força o modelo a olhar regiões distintas
    'gauss_blur': 0.2,       # simula degradação de qualidade / compressão
    'gauss_noise_std': 0.02, # simula artefatos de compressão JPEG (prob 0.2, std 0.02)
}

# Proporções do split train/val/test (aplicadas sobre imagens originais)
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Threshold de decisão para classificação (< 0.5 favorece recall)
DECISION_THRESHOLD = 0.4

# Configurações de fine-tuning específicas por modelo (apenas ResNet-50 e DINOv2)
FINETUNE_CONFIGS = {
    'resnet50': {
        # Fase 1: treinar só o classificador (backbone ImageNet congelado)
        'phase1_epochs': 8,
        'phase1_lr': 5e-4,
        # Fase 2: fine-tuning completo com lr diferenciado
        'phase2_epochs': 40,
        'phase2_backbone_lr': 2e-5,       # revertido: 5e-5 era agressivo demais, modelo colapsou
        'phase2_classifier_lr': 2e-4,
        'warmup_epochs': 3,
        'weight_decay': 1e-4,             # revertido: 1e-3 + class_weights causou colapso para forged
        'label_smoothing': 0.1,
        'early_stopping_patience': 12,
    },
    'dinov2': {
        'phase1_epochs': 12,
        'phase1_lr': 5e-4,
        'phase2_epochs': 35,           # +15: mais tempo para fine-tuning com resolução 384
        'phase2_backbone_lr': 5e-6,    # 5x maior: backbone precisa adaptar mais para 384px
        'phase2_classifier_lr': 5e-5,  # 2.5x maior: alinhado com aumento do backbone LR
        'warmup_epochs': 4,
        'weight_decay': 0.05,
        'label_smoothing': 0.05,       # reduzido de 0.1: suavizava demais o sinal de forged
        'early_stopping_patience': 12, # 2x mais paciente: com LR baixo, melhora é gradual
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
