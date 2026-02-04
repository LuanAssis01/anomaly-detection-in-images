"""
Interface de demonstração para o laboratório de detecção de falsificações
"""
import os
import sys
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from configs.config import *
from models import CNNModel, ViTModel, CvTModel, DINOv2Model

class ForgeryDetector:
    """Interface para detecção de falsificações"""
    
    def __init__(self, model_type='cvt21'):
        self.device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type
        
        # Carregar modelo
        checkpoint_path = os.path.join(CHECKPOINTS_DIR, f'{model_type}_best.pth')
        
        if not os.path.exists(checkpoint_path):
            print(f"⚠ Modelo {model_type} não encontrado. Treine-o primeiro com:")
            print(f"   python train.py --model {model_type}")
            self.model = None
            return
        
        print(f"Carregando modelo {model_type}...")
        
        # Criar modelo
        if model_type in ['resnet50', 'resnet101']:
            self.model = CNNModel(
                num_classes=NUM_CLASSES,
                model_name=MODEL_CONFIGS[model_type]['model_name'],
                pretrained=False
            )
        elif model_type == 'vit':
            self.model = ViTModel(
                model_name=MODEL_CONFIGS['vit']['model_name'],
                num_classes=NUM_CLASSES
            )
        elif model_type in ['cvt13', 'cvt21', 'cvt_w24']:
            self.model = CvTModel(
                model_name=MODEL_CONFIGS[model_type]['model_name'],
                num_classes=NUM_CLASSES
            )
        elif model_type == 'dinov2':
            self.model = DINOv2Model(
                model_name=MODEL_CONFIGS['dinov2']['model_name'],
                num_classes=NUM_CLASSES
            )
        
        # Carregar pesos
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Transform
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f"✓ Modelo {model_type} carregado com sucesso!")
        print(f"  Acurácia de validação: {checkpoint.get('val_acc', 0):.2f}%")
    
    def predict(self, image_path):
        """Faz predição em uma imagem"""
        if self.model is None:
            return None
        
        # Carregar imagem
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Predição
        with torch.no_grad():
            output = self.model(image_tensor)
            probs = torch.softmax(output, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_class].item()
        
        result = {
            'class': 'Forged' if pred_class == 1 else 'Authentic',
            'confidence': confidence * 100,
            'prob_authentic': probs[0, 0].item() * 100,
            'prob_forged': probs[0, 1].item() * 100
        }
        
        return result
    
    def visualize_prediction(self, image_path):
        """Visualiza predição"""
        result = self.predict(image_path)
        
        if result is None:
            return
        
        # Mostrar imagem e resultado
        image = Image.open(image_path)
        
        plt.figure(figsize=(10, 5))
        
        # Imagem
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title('Imagem')
        plt.axis('off')
        
        # Probabilidades
        plt.subplot(1, 2, 2)
        classes = ['Authentic', 'Forged']
        probs = [result['prob_authentic'], result['prob_forged']]
        colors = ['green', 'red']
        plt.bar(classes, probs, color=colors, alpha=0.7)
        plt.ylabel('Probabilidade (%)')
        plt.title(f"Predição: {result['class']}\nConfiança: {result['confidence']:.2f}%")
        plt.ylim([0, 100])
        
        plt.tight_layout()
        plt.show()
        
        print(f"\nResultado:")
        print(f"  Classe: {result['class']}")
        print(f"  Confiança: {result['confidence']:.2f}%")
        print(f"  Prob. Autêntica: {result['prob_authentic']:.2f}%")
        print(f"  Prob. Falsificada: {result['prob_forged']:.2f}%")


def demo():
    """Demonstração do sistema"""
    print("="*60)
    print("DETECTOR DE FALSIFICAÇÕES EM IMAGENS")
    print("="*60)
    
    # Testar com imagem do dataset
    authentic_dir = os.path.join(TRAIN_DIR, 'authentic')
    forged_dir = os.path.join(TRAIN_DIR, 'forged')
    
    if os.path.exists(authentic_dir):
        authentic_images = [f for f in os.listdir(authentic_dir) if f.endswith('.png')][:3]
    else:
        authentic_images = []
    
    if os.path.exists(forged_dir):
        forged_images = [f for f in os.listdir(forged_dir) if f.endswith('.png')][:3]
    else:
        forged_images = []
    
    # Criar detector (usando CvT-21 por padrão)
    detector = ForgeryDetector('cvt21')
    
    if detector.model is None:
        print("\n⚠ Nenhum modelo treinado encontrado.")
        print("Execute primeiro: python train.py --model cvt21")
        return
    
    print("\nTestando imagens autênticas:")
    for img_name in authentic_images:
        img_path = os.path.join(authentic_dir, img_name)
        result = detector.predict(img_path)
        print(f"  {img_name}: {result['class']} ({result['confidence']:.1f}%)")
    
    print("\nTestando imagens falsificadas:")
    for img_name in forged_images:
        img_path = os.path.join(forged_dir, img_name)
        result = detector.predict(img_path)
        print(f"  {img_name}: {result['class']} ({result['confidence']:.1f}%)")
    
    print("\n" + "="*60)


if __name__ == '__main__':
    demo()
