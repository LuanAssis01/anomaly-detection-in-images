"""
Funções de visualização para resultados
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional
import torch
from PIL import Image

def plot_training_history(history: Dict[str, List[float]], save_path: Optional[str] = None):
    """
    Plota curvas de loss e acurácia durante o treinamento
    
    Args:
        history: Dicionário com 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        save_path: Caminho para salvar a figura
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss
    if 'train_loss' in history and 'val_loss' in history:
        axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
        axes[0].plot(history['val_loss'], label='Validation Loss', marker='s')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    if 'train_acc' in history and 'val_acc' in history:
        axes[1].plot(history['train_acc'], label='Train Accuracy', marker='o')
        axes[1].plot(history['val_acc'], label='Validation Accuracy', marker='s')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], save_path: Optional[str] = None):
    """
    Plota matriz de confusão
    
    Args:
        cm: Matriz de confusão
        class_names: Nomes das classes
        save_path: Caminho para salvar
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Matriz de Confusão')
    plt.ylabel('Real')
    plt.xlabel('Predito')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_metrics_comparison(results: Dict[str, Dict[str, float]], save_path: Optional[str] = None):
    """
    Compara métricas entre diferentes modelos
    
    Args:
        results: {model_name: {metric: value}}
        save_path: Caminho para salvar
    """
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
    
    models = list(results.keys())
    n_models = len(models)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(metrics_to_plot))
    width = 0.8 / n_models
    
    for i, model in enumerate(models):
        values = [results[model].get(metric, 0) * 100 for metric in metrics_to_plot]
        ax.bar(x + i * width, values, width, label=model)
    
    ax.set_xlabel('Métricas')
    ax.set_ylabel('Score (%)')
    ax.set_title('Comparação de Métricas entre Modelos')
    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels([m.capitalize().replace('_', ' ') for m in metrics_to_plot])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 105])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def visualize_predictions(
    images: torch.Tensor,
    labels: torch.Tensor,
    predictions: torch.Tensor,
    num_images: int = 8,
    save_path: Optional[str] = None
):
    """
    Visualiza predições do modelo
    
    Args:
        images: Tensor de imagens [B, 3, H, W]
        labels: Labels verdadeiros
        predictions: Labels preditos
        num_images: Número de imagens a mostrar
        save_path: Caminho para salvar
    """
    num_images = min(num_images, len(images))
    
    fig, axes = plt.subplots(2, num_images // 2, figsize=(15, 6))
    axes = axes.flatten()
    
    # Desnormalizar imagens (mean e std do ImageNet)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    for i in range(num_images):
        img = images[i].cpu() * std + mean
        img = img.permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        
        axes[i].imshow(img)
        
        true_label = 'Forged' if labels[i] == 1 else 'Authentic'
        pred_label = 'Forged' if predictions[i] == 1 else 'Authentic'
        
        color = 'green' if labels[i] == predictions[i] else 'red'
        axes[i].set_title(f'Real: {true_label}\nPred: {pred_label}', color=color)
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def visualize_results(images, masks_true, masks_pred, num_samples=4):
    """
    Visualiza imagens, máscaras verdadeiras e preditas
    
    Args:
        images: Tensor de imagens
        masks_true: Máscaras verdadeiras
        masks_pred: Máscaras preditas
        num_samples: Número de amostras a visualizar
    """
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
    
    for i in range(num_samples):
        # Imagem original
        axes[i, 0].imshow(images[i])
        axes[i, 0].set_title('Imagem Original')
        axes[i, 0].axis('off')
        
        # Máscara verdadeira
        axes[i, 1].imshow(masks_true[i], cmap='gray')
        axes[i, 1].set_title('Máscara Verdadeira')
        axes[i, 1].axis('off')
        
        # Máscara predita
        axes[i, 2].imshow(masks_pred[i], cmap='gray')
        axes[i, 2].set_title('Máscara Predita')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.show()
