"""
DINOv2 e DINOv3 para detecção de falsificações
"""
import torch
import torch.nn as nn
from transformers import Dinov2Model, AutoModel

class DINOv2Model(nn.Module):
    """
    DINOv2: Self-supervised Vision Transformer
    Treinado com técnicas de destilação
    """
    def __init__(self, model_name: str = 'facebook/dinov2-base', num_classes: int = 2):
        super(DINOv2Model, self).__init__()

        # Carregar modelo pré-treinado
        self.dino = Dinov2Model.from_pretrained(model_name)

        # Dimensão de saída do DINOv2
        self.hidden_size = self.dino.config.hidden_size

        # Cabeça de classificação (dropout reduzido para não sub-utilizar features)
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(768, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(384, num_classes)
        )

    def forward(self, pixel_values):
        """
        Args:
            pixel_values: Tensor [batch_size, 3, H, W] — default IMAGE_SIZE=384
        Returns:
            logits: Tensor [batch_size, num_classes]
        """
        # Passar pelo DINOv2 (interpolate_pos_encoding permite resolução diferente de 224)
        outputs = self.dino(pixel_values=pixel_values, interpolate_pos_encoding=True)

        # Usar o token [CLS]
        cls_token = outputs.last_hidden_state[:, 0, :]

        # Classificação
        logits = self.classifier(cls_token)

        return logits

    def get_model_name(self):
        return "DINOv2"

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze_backbone(self):
        """Congela os pesos do DINOv2"""
        for param in self.dino.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Descongela o backbone"""
        for param in self.dino.parameters():
            param.requires_grad = True

    def get_param_groups(self, backbone_lr, classifier_lr):
        """Retorna grupos de parâmetros com learning rates diferenciados"""
        return [
            {'params': self.dino.parameters(), 'lr': backbone_lr},
            {'params': self.classifier.parameters(), 'lr': classifier_lr},
        ]


class DINOv3Model(nn.Module):
    """
    DINOv3: Self-supervised Vision Transformer (geração seguinte ao DINOv2)
    Suporta variantes: small (ViT-S/16), base (ViT-B/16), large (ViT-L/16)

    Diferenças em relação ao DINOv2:
    - patch_size=16 (DINOv2 usa 14)
    - Treinado no LVD-1689M (dataset maior)
    - Carregado via AutoModel para compatibilidade com diferentes variantes
    """
    def __init__(self, model_name: str = 'facebook/dinov3-vitb16-pretrain-lvd1689m', num_classes: int = 2):
        super(DINOv3Model, self).__init__()

        # Carregar backbone via AutoModel (suporta small/base/large)
        self.backbone = AutoModel.from_pretrained(model_name)

        # Dimensão de saída (384=small, 768=base, 1024=large)
        self.hidden_size = self.backbone.config.hidden_size

        # Cabeça de classificação escalada pelo hidden_size
        mid_size = self.hidden_size
        small_size = self.hidden_size // 2

        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, mid_size),
            nn.LayerNorm(mid_size),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(mid_size, small_size),
            nn.LayerNorm(small_size),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(small_size, num_classes)
        )

    def forward(self, pixel_values):
        """
        Args:
            pixel_values: Tensor [batch_size, 3, H, W]
        Returns:
            logits: Tensor [batch_size, num_classes]
        """
        # interpolate_pos_encoding=True permite resoluções diferentes do treino
        outputs = self.backbone(pixel_values=pixel_values, interpolate_pos_encoding=True)

        # Token [CLS]
        cls_token = outputs.last_hidden_state[:, 0, :]

        return self.classifier(cls_token)

    def get_model_name(self):
        return "DINOv3"

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze_backbone(self):
        """Congela os pesos do backbone DINOv3"""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Descongela o backbone"""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def get_param_groups(self, backbone_lr, classifier_lr):
        """Retorna grupos de parâmetros com learning rates diferenciados"""
        return [
            {'params': self.backbone.parameters(), 'lr': backbone_lr},
            {'params': self.classifier.parameters(), 'lr': classifier_lr},
        ]
