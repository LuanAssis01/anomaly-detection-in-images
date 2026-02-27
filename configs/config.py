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
SUPPLEMENTAL_MASKS_DIR = os.path.join(DATA_DIR, 'supplemental_masks')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
CHECKPOINTS_DIR = os.path.join(BASE_DIR, 'checkpoints')

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

# Configurações dos modelos.
MODEL_CONFIGS = {
    'resnet50': {
        'name': 'ResNet-50',
        'model_name': 'resnet50',
        'num_classes': NUM_CLASSES
    },
    'vit': {
        'name': 'Vision Transformer',
        'model_name': 'google/vit-base-patch16-384',
        'num_classes': NUM_CLASSES
    },
    'cvt13': {
        'name': 'CvT-13',
        'model_name': 'microsoft/cvt-13',
        'num_classes': NUM_CLASSES
    },
    'cvt21': {
        'name': 'CvT-21',
        'model_name': 'microsoft/cvt-21',
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

# Threshold de decisão para classificação (< 0.5 favorece recall)
DECISION_THRESHOLD = 0.4

# Configurações de fine-tuning específicas por modelo
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
    'vit': {
        # Fase 1: treinar só a cabeça (backbone congelado)
        'phase1_epochs': 10,
        'phase1_lr': 5e-4,                # reduzido de 1e-3: LR alto destruía features do backbone
        # Fase 2: fine-tuning completo com lr diferenciado
        'phase2_epochs': 40,
        'phase2_backbone_lr': 1e-5,       # 5x maior: backbone precisa re-adaptar após fase 1 agressiva
        'phase2_classifier_lr': 1e-4,
        'warmup_epochs': 5,
        'weight_decay': 0.05,             # 5x maior: mais regularização para evitar overfit
        'label_smoothing': 0.1,           # adicionado: sem regularização na loss antes
        'early_stopping_patience': 15,    # adicionado: antes era disabled (999), causava overfit
    },
    'cvt13': {
        'phase1_epochs': 10,
        'phase1_lr': 5e-4,                # reduzido de 1e-3: mesma razão do ViT
        'phase2_epochs': 40,
        'phase2_backbone_lr': 1e-5,       # 2x maior: backbone precisa re-adaptar
        'phase2_classifier_lr': 1e-4,
        'warmup_epochs': 5,
        'weight_decay': 0.05,             # 5x maior: mais regularização
        'label_smoothing': 0.1,           # adicionado
        'early_stopping_patience': 15,    # adicionado: antes era disabled
    },
    'cvt21': {
        'phase1_epochs': 10,
        'phase1_lr': 5e-4,                # reduzido de 1e-3
        'phase2_epochs': 40,
        'phase2_backbone_lr': 1e-5,       # 2x maior
        'phase2_classifier_lr': 1e-4,
        'warmup_epochs': 5,
        'weight_decay': 0.05,             # 5x maior
        'label_smoothing': 0.1,           # adicionado
        'early_stopping_patience': 15,    # adicionado
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

# Random seed para reprodutibilidade
RANDOM_SEED = 42
