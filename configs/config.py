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
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4
DEVICE = 'cuda'  # 'cuda' ou 'cpu'
NUM_WORKERS = 4  # Workers paralelos para carregar dados (0 = sequencial, lento)

# Otimizações de performance (GPU)
USE_AMP = True                    # Mixed Precision (FP16) — ~2x mais rápido com Tensor Cores
GRADIENT_ACCUMULATION_STEPS = 1   # Acumular gradientes (>1 simula batch maior sem usar mais VRAM)
USE_COMPILE = False                # torch.compile() — funde operações para execução mais rápida

# Parâmetros das imagens
IMAGE_SIZE = 224
NUM_CLASSES = 2  # authentic vs forged

# Configurações dos modelos.
MODEL_CONFIGS = {
    'resnet50': {
        'name': 'ResNet-50',
        'model_name': 'resnet50',
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
}

# Configurações de fine-tuning específicas por modelo
FINETUNE_CONFIGS = {
    'resnet50': {
        # Fase 1: treinar só o classificador (backbone ImageNet congelado)
        'phase1_epochs': 8,               # reduzido: classificador sozinho satura rápido no ResNet
        'phase1_lr': 5e-4,               # mais suave: 1e-3 era instável com o classificador pequeno
        # Fase 2: fine-tuning completo com lr diferenciado
        'phase2_epochs': 30,              # mais tempo: ResNet precisa adaptar features ImageNet
        'phase2_backbone_lr': 2e-5,       # 4x maior: ResNet precisa de mais adaptação que DINOv2
        'phase2_classifier_lr': 2e-4,     # 2.5x maior: classificador convergiu pouco na fase 1
        'warmup_epochs': 3,               # warmup curto: ResNet já tem boas features iniciais
        'weight_decay': 1e-4,             # reduzido: 5e-4 + dropout 0.3 era sobre-regularização
        'label_smoothing': 0.05,          # reduzido: 0.1 atrapalhava a convergência com dataset pequeno
        'early_stopping_patience': 10,    # mais paciente: cosine scheduler melhora no final
    },
    'vit': {
        # Fase 1: treinar só a cabeça (backbone congelado)
        'phase1_epochs': 10,
        'phase1_lr': 1e-3,
        # Fase 2: fine-tuning completo com lr diferenciado
        'phase2_epochs': 40,
        'phase2_backbone_lr': 2e-6,
        'phase2_classifier_lr': 1e-4,
        'warmup_epochs': 5,
        'weight_decay': 0.01,
    },
    'cvt13': {
        'phase1_epochs': 10,
        'phase1_lr': 1e-3,
        'phase2_epochs': 40,
        'phase2_backbone_lr': 5e-6,   # CvT tem convoluções, responde melhor a fine-tuning
        'phase2_classifier_lr': 1e-4,
        'warmup_epochs': 5,
        'weight_decay': 0.01,
    },
    'cvt21': {
        'phase1_epochs': 10,
        'phase1_lr': 1e-3,
        'phase2_epochs': 40,
        'phase2_backbone_lr': 5e-6,
        'phase2_classifier_lr': 1e-4,
        'warmup_epochs': 5,
        'weight_decay': 0.01,
    },
    'dinov2': {
        'phase1_epochs': 12,
        'phase1_lr': 5e-4,             # suavizado: classificador precisa de convergência estável
        'phase2_epochs': 20,           # reduzido drasticamente: val_loss divergia a partir da época 17
        'phase2_backbone_lr': 1e-6,    # DINOv2 tem features muito fortes, lr ultra-baixo
        'phase2_classifier_lr': 2e-5,  # reduzido: 5e-5 era agressivo demais para DINOv2
        'warmup_epochs': 4,
        'weight_decay': 0.05,          # 5x mais forte: 0.01→0.05, backbone de 86M parâmetros
        'label_smoothing': 0.1,
        'early_stopping_patience': 6,  # DINOv2 overfita rápido, paciência menor
    },
}

# Random seed para reprodutibilidade
RANDOM_SEED = 42
