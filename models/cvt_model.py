"""
Convolutional Vision Transformer (CvT) para detecção de falsificações
"""
import torch
import torch.nn as nn
from transformers import CvtModel, CvtConfig

class CvTModel(nn.Module):
    """
    Convolutional Vision Transformer
    Combina convoluções com self-attention
    Suporta: cvt-13, cvt-21, cvt-w24
    """
    def __init__(self, model_name: str = 'microsoft/cvt-21', num_classes: int = 2):
        super(CvTModel, self).__init__()
        
        self.model_variant = model_name.split('/')[-1]  # 'cvt-13', 'cvt-21', ou 'cvt-w24'
        
        # Carregar modelo pré-treinado
        self.cvt = CvtModel.from_pretrained(model_name)
        
        # Obter dimensão de saída
        # CvT usa pooling nas features da última camada
        self.hidden_size = self.cvt.config.embed_dim[-1]
        
        # Cabeça de classificação
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
            pixel_values: Tensor [batch_size, 3, 224, 224]
        Returns:
            logits: Tensor [batch_size, num_classes]
        """
        # Passar pelo CvT
        outputs = self.cvt(pixel_values=pixel_values)
        
        # CvT retorna BaseModelOutputWithCLSToken
        # Usar o cls_token_value que contém as features globais
        pooled_output = outputs.cls_token_value
        
        # Garantir que temos o formato correto [batch_size, hidden_size]
        if pooled_output.dim() == 3:
            pooled_output = pooled_output.squeeze(1)
        
        # Classificação
        logits = self.classifier(pooled_output)
        
        return logits
    
    def get_model_name(self):
        return f"CvT-{self.model_variant.split('-')[-1].upper()}"
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def freeze_backbone(self):
        """Congela os pesos do CvT"""
        for param in self.cvt.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Descongela o backbone"""
        for param in self.cvt.parameters():
            param.requires_grad = True
    
    def get_param_groups(self, backbone_lr, classifier_lr):
        """Retorna grupos de parâmetros com learning rates diferenciados"""
        return [
            {'params': self.cvt.parameters(), 'lr': backbone_lr},
            {'params': self.classifier.parameters(), 'lr': classifier_lr},
        ]
