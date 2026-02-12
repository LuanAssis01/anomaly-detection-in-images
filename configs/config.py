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
BATCH_SIZE = 16
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
DEVICE = 'cuda'  # 'cuda' ou 'cpu'
NUM_WORKERS = 4  # Workers paralelos para carregar dados (0 = sequencial, lento)

# Otimizações de performance (GPU)
USE_AMP = True                    # Mixed Precision (FP16) — ~2x mais rápido com Tensor Cores
GRADIENT_ACCUMULATION_STEPS = 1   # Acumular gradientes (>1 simula batch maior sem usar mais VRAM)
USE_COMPILE = True                # torch.compile() — funde operações para execução mais rápida

# Parâmetros das imagens
IMAGE_SIZE = 224
NUM_CLASSES = 2  # authentic vs forged

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
    'vit': {
        'name': 'Vision Transformer',
        'model_name': 'google/vit-base-patch16-224',
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
    'cvt_w24': {
        'name': 'CvT-W24',
        'model_name': 'microsoft/cvt-w24-384-22k',
        'num_classes': NUM_CLASSES,
        'image_size': 384  # Esse modelo foi pré-treinado com 384x384
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
    'brightness': 0.2,
    'contrast': 0.2
}

# Random seed para reprodutibilidade
RANDOM_SEED = 42
