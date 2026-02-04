"""
Vision Transformer (ViT) para detecção de falsificações
"""
import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig

class ViTModel(nn.Module):
    """
    Vision Transformer adaptado para classificação binária
    """
    def __init__(self, model_name: str = 'google/vit-base-patch16-224', num_classes: int = 2):
        super(ViTModel, self).__init__()
        
        # Carregar modelo pré-treinado
        self.vit = ViTModel.from_pretrained(model_name)
        
        # Obter dimensão de saída do ViT
        self.hidden_size = self.vit.config.hidden_size
        
        # Cabeça de classificação customizada
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, pixel_values):
        """
        Args:
            pixel_values: Tensor de imagens [batch_size, 3, 224, 224]
        Returns:
            logits: Tensor [batch_size, num_classes]
        """
        # Passar pela ViT
        outputs = self.vit(pixel_values=pixel_values)
        
        # Pegar o token [CLS] (primeiro token)
        cls_token = outputs.last_hidden_state[:, 0, :]
        
        # Classificação
        logits = self.classifier(cls_token)
        
        return logits
    
    def get_model_name(self):
        return "Vision-Transformer"
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def freeze_backbone(self):
        """Congela os pesos do ViT, treina apenas o classificador"""
        for param in self.vit.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Descongela o backbone para fine-tuning completo"""
        for param in self.vit.parameters():
            param.requires_grad = True
