"""
Métricas de avaliação para detecção de falsificações
"""
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, roc_auc_score, 
    average_precision_score
)
from typing import Dict, Optional

def calculate_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    y_prob: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Calcula métricas de classificação
    
    Args:
        y_true: Labels verdadeiros
        y_pred: Labels preditos
        y_prob: Probabilidades da classe positiva (para AUC)
    
    Returns:
        Dicionário com as métricas
    """
    metrics = {}
    
    # Métricas básicas
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
    
    # Matriz de confusão
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    
    # True Negatives, False Positives, False Negatives, True Positives
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['true_positives'] = int(tp)
        
        # Specificity
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['specificity'] = specificity
    
    # AUC-ROC e AUC-PR se probabilidades foram fornecidas
    if y_prob is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
            metrics['pr_auc'] = average_precision_score(y_true, y_prob)
        except ValueError:
            # Quando só há uma classe presente
            metrics['roc_auc'] = 0.0
            metrics['pr_auc'] = 0.0
    
    return metrics

def calculate_iou(pred_mask: np.ndarray, true_mask: np.ndarray, threshold: float = 0.5) -> float:
    """
    Calcula Intersection over Union (IoU) para máscaras de segmentação
    
    Args:
        pred_mask: Máscara predita [H, W]
        true_mask: Máscara verdadeira [H, W]
        threshold: Threshold para binarizar a predição
    
    Returns:
        IoU score
    """
    # Binarizar máscaras
    pred_binary = (pred_mask > threshold).astype(np.uint8)
    true_binary = (true_mask > threshold).astype(np.uint8)
    
    # Calcular interseção e união
    intersection = np.logical_and(pred_binary, true_binary).sum()
    union = np.logical_or(pred_binary, true_binary).sum()
    
    if union == 0:
        return 0.0
    
    return intersection / union

def calculate_pixel_accuracy(pred_mask: np.ndarray, true_mask: np.ndarray, threshold: float = 0.5) -> float:
    """
    Calcula acurácia pixel a pixel
    
    Args:
        pred_mask: Máscara predita [H, W]
        true_mask: Máscara verdadeira [H, W]
        threshold: Threshold para binarizar
    
    Returns:
        Acurácia
    """
    pred_binary = (pred_mask > threshold).astype(np.uint8)
    true_binary = (true_mask > threshold).astype(np.uint8)
    
    correct = (pred_binary == true_binary).sum()
    total = true_binary.size
    
    return correct / total

class AverageMeter:
    """Calcula e armazena a média e o valor atual"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
