"""
Script de teste rápido para verificar instalação e modelos
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Testa se todas as bibliotecas estão instaladas"""
    print("Testando imports...")

    try:
        import torch
        print(f"  PyTorch {torch.__version__}")
        print(f"  CUDA disponível: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("  PyTorch não instalado")

    try:
        import torchvision
        print(f"  TorchVision {torchvision.__version__}")
    except ImportError:
        print("  TorchVision não instalado")

    try:
        import transformers
        print(f"  Transformers {transformers.__version__}")
    except ImportError:
        print("  Transformers não instalado")

    try:
        import numpy as np
        print(f"  NumPy {np.__version__}")
    except ImportError:
        print("  NumPy não instalado")

    try:
        import sklearn
        print(f"  Scikit-learn {sklearn.__version__}")
    except ImportError:
        print("  Scikit-learn não instalado")

    try:
        import matplotlib
        print(f"  Matplotlib {matplotlib.__version__}")
    except ImportError:
        print("  Matplotlib não instalado")

    try:
        from PIL import Image
        print(f"  Pillow instalado")
    except ImportError:
        print("  Pillow não instalado")

def test_models():
    """Testa se os modelos podem ser carregados"""
    print("\nTestando modelos...")

    try:
        import torchvision.models as models
        model = models.resnet50(weights=None)
        params = sum(p.numel() for p in model.parameters())
        print(f"  ResNet-50 carregado ({params:,} parâmetros)")
        del model
    except Exception as e:
        print(f"  Erro ao carregar ResNet-50: {e}")

    try:
        from transformers import Dinov2Model
        model = Dinov2Model.from_pretrained('facebook/dinov2-base')
        params = sum(p.numel() for p in model.parameters())
        print(f"  DINOv2 carregado ({params:,} parâmetros)")
        del model
    except Exception as e:
        print(f"  Erro ao carregar DINOv2: {e}")

def test_dataset():
    """Verifica o dataset"""
    print("\nVerificando dataset...")

    from configs.config import SCENARIOS

    for scenario, cfg in SCENARIOS.items():
        train_dir = cfg['train_dir']
        authentic_dir = os.path.join(train_dir, 'authentic')
        forged_dir = os.path.join(train_dir, 'forged')

        print(f"\n  Cenário: {scenario}")
        if os.path.exists(authentic_dir):
            n = len([f for f in os.listdir(authentic_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            print(f"    Autênticas: {n}")
        else:
            print(f"    Pasta autênticas não encontrada")

        if os.path.exists(forged_dir):
            n = len([f for f in os.listdir(forged_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            print(f"    Forjadas: {n}")
        else:
            print(f"    Pasta forjadas não encontrada")

def main():
    print("=" * 60)
    print("TESTE DO AMBIENTE DE DESENVOLVIMENTO")
    print("=" * 60 + "\n")

    test_imports()
    test_models()
    test_dataset()

    print("\n" + "=" * 60)
    print("Testes concluídos!")
    print("=" * 60)

if __name__ == '__main__':
    main()
