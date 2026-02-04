"""
Modelo ResNet para detecção de imagens forjadas
"""
import torch
import torch.nn as nn
from torchvision import models

class CNNModel(nn.Module):
    """
    ResNet pré-treinado para classificação de imagens
    Serve como baseline CNN para comparação com Transformers
    """
    def __init__(self, num_classes: int = 2, model_name: str = 'resnet50', pretrained: bool = True):
        super(CNNModel, self).__init__()
        
        self.model_name = model_name
        
        # Carregar ResNet pré-treinado
        if model_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            feature_dim = 2048
        elif model_name == 'resnet101':
            self.backbone = models.resnet101(pretrained=pretrained)
            feature_dim = 2048
        elif model_name == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
            feature_dim = 512
        else:
            raise ValueError(f"Modelo {model_name} não suportado. Use 'resnet34', 'resnet50' ou 'resnet101'.")
        
        # Remover a última camada FC
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Cabeça de classificação customizada
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        # Extrair features com ResNet
        features = self.backbone(x)
        
        # Classificação
        logits = self.classifier(features)
        
        return logits
    
    def get_model_name(self):
        return f"ResNet-{self.model_name.replace('resnet', '')}"
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def freeze_backbone(self):
        """Congela os pesos do ResNet"""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Descongela o backbone"""
        for param in self.backbone.parameters():
            param.requires_grad = True
