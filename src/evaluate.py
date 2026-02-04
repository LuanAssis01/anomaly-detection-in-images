"""
Script de avaliação e testes dos modelos treinados
"""
import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import argparse

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from configs.config import *
from utils.dataset import ForgeryDataset, get_transforms, custom_collate_fn
from utils.metrics import calculate_metrics
from utils.visualization import plot_confusion_matrix, visualize_predictions
from models import CNNModel, ViTModel, CvTModel, DINOv2Model

def load_model(model_type: str, checkpoint_path: str, device):
    """Carrega modelo treinado"""
    print(f"Carregando modelo {model_type}...")
    
    # Criar modelo
    if model_type in ['resnet50', 'resnet101']:
        model = CNNModel(
            num_classes=NUM_CLASSES,
            model_name=MODEL_CONFIGS[model_type]['model_name'],
            pretrained=False  # Vamos carregar pesos treinados
        )
    elif model_type == 'vit':
        model = ViTModel(
            model_name=MODEL_CONFIGS['vit']['model_name'],
            num_classes=NUM_CLASSES
        )
    elif model_type in ['cvt13', 'cvt21', 'cvt_w24']:
        model = CvTModel(
            model_name=MODEL_CONFIGS[model_type]['model_name'],
            num_classes=NUM_CLASSES
        )
    elif model_type == 'dinov2':
        model = DINOv2Model(
            model_name=MODEL_CONFIGS['dinov2']['model_name'],
            num_classes=NUM_CLASSES
        )
    else:
        raise ValueError(f"Modelo desconhecido: {model_type}")
    
    # Carregar pesos
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✓ Modelo carregado (acurácia de treino: {checkpoint.get('val_acc', 0):.2f}%)")
    
    return model

def evaluate_model(model, dataloader, device):
    """Avalia modelo no dataset"""
    model.eval()
    
    all_labels = []
    all_predictions = []
    all_probs = []
    all_images = []
    
    with torch.no_grad():
        for images, labels, _ in tqdm(dataloader, desc='Avaliando'):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            _, predicted = torch.max(outputs.data, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            # Guardar algumas imagens para visualização
            if len(all_images) < 16:
                all_images.extend(images.cpu())
    
    # Calcular métricas
    metrics = calculate_metrics(
        np.array(all_labels),
        np.array(all_predictions),
        np.array(all_probs)
    )
    
    return metrics, all_labels, all_predictions, all_images[:16]

def print_metrics(model_name: str, metrics: dict):
    """Imprime métricas de forma formatada"""
    print(f"\n{'='*60}")
    print(f"Resultados - {model_name}")
    print(f"{'='*60}")
    print(f"Accuracy:    {metrics['accuracy']*100:.2f}%")
    print(f"Precision:   {metrics['precision']:.4f}")
    print(f"Recall:      {metrics['recall']:.4f}")
    print(f"F1-Score:    {metrics['f1_score']:.4f}")
    print(f"Specificity: {metrics.get('specificity', 0):.4f}")
    
    if 'roc_auc' in metrics:
        print(f"ROC-AUC:     {metrics['roc_auc']:.4f}")
        print(f"PR-AUC:      {metrics['pr_auc']:.4f}")
    
    print(f"\nMatriz de Confusão:")
    cm = np.array(metrics['confusion_matrix'])
    print(f"  TN: {metrics.get('true_negatives', 0):4d}  FP: {metrics.get('false_positives', 0):4d}")
    print(f"  FN: {metrics.get('false_negatives', 0):4d}  TP: {metrics.get('true_positives', 0):4d}")
    print(f"{'='*60}\n")

def main():
    parser = argparse.ArgumentParser(description='Avaliar modelos treinados')
    parser.add_argument('--model', type=str, default='all',
                      choices=['all', 'resnet50', 'resnet101', 'vit', 'cvt13', 'cvt21', 'cvt_w24', 'dinov2'],
                      help='Modelo a avaliar')
    parser.add_argument('--visualize', action='store_true',
                      help='Mostrar visualizações')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Dataset de teste
    print("\nCarregando dataset de teste...")
    # Como não temos test labels, vamos usar parte do train como teste
    from torch.utils.data import random_split
    
    full_dataset = ForgeryDataset(
        root_dir=TRAIN_DIR,
        masks_dir=TRAIN_MASKS_DIR,
        transform=get_transforms(IMAGE_SIZE, mode='val'),
        mode='test'
    )
    
    # 80% treino, 20% teste
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    _, test_dataset = random_split(
        full_dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED)
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=custom_collate_fn
    )
    
    print(f"Samples de teste: {len(test_dataset)}")
    
    # Determinar modelos a avaliar
    if args.model == 'all':
        models_to_eval = ['resnet50', 'vit', 'cvt13', 'cvt21', 'cvt_w24', 'dinov2']
    else:
        models_to_eval = [args.model]
    
    all_results = {}
    
    # Avaliar cada modelo
    for model_type in models_to_eval:
        checkpoint_path = os.path.join(CHECKPOINTS_DIR, f'{model_type}_best.pth')
        
        if not os.path.exists(checkpoint_path):
            print(f"\n⚠ Checkpoint não encontrado para {model_type}: {checkpoint_path}")
            continue
        
        # Carregar e avaliar
        model = load_model(model_type, checkpoint_path, device)
        metrics, labels, predictions, images = evaluate_model(model, test_loader, device)
        
        # Salvar resultados
        all_results[model_type] = metrics
        
        # Imprimir
        print_metrics(MODEL_CONFIGS[model_type]['name'], metrics)
        
        # Salvar métricas em JSON
        results_path = os.path.join(RESULTS_DIR, f'{model_type}_test_results.json')
        with open(results_path, 'w') as f:
            # Remover confusion_matrix para JSON
            metrics_save = {k: v for k, v in metrics.items() if k != 'confusion_matrix'}
            json.dump(metrics_save, f, indent=2)
        
        # Visualizações
        if args.visualize:
            # Matriz de confusão
            cm = np.array(metrics['confusion_matrix'])
            plot_confusion_matrix(
                cm, 
                ['Authentic', 'Forged'],
                save_path=os.path.join(RESULTS_DIR, f'{model_type}_confusion_matrix.png')
            )
            
            # Predições
            visualize_predictions(
                torch.stack(images[:8]),
                torch.tensor(labels[:8]),
                torch.tensor(predictions[:8]),
                num_images=8,
                save_path=os.path.join(RESULTS_DIR, f'{model_type}_predictions.png')
            )
    
    # Comparação geral
    if len(all_results) > 1:
        print("\n" + "="*60)
        print("COMPARAÇÃO GERAL DOS MODELOS")
        print("="*60)
        
        # Tabela comparativa
        print(f"\n{'Modelo':<25} {'Acc':<8} {'Prec':<8} {'Rec':<8} {'F1':<8}")
        print("-" * 60)
        for model_type, metrics in all_results.items():
            model_name = MODEL_CONFIGS[model_type]['name']
            print(f"{model_name:<25} "
                  f"{metrics['accuracy']*100:<8.2f} "
                  f"{metrics['precision']:<8.4f} "
                  f"{metrics['recall']:<8.4f} "
                  f"{metrics['f1_score']:<8.4f}")
        print("="*60 + "\n")
        
        # Salvar comparação
        comparison_path = os.path.join(RESULTS_DIR, 'model_comparison.json')
        with open(comparison_path, 'w') as f:
            comparison = {}
            for model_type, metrics in all_results.items():
                comparison[model_type] = {
                    'model_name': MODEL_CONFIGS[model_type]['name'],
                    'accuracy': metrics['accuracy'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1_score': metrics['f1_score']
                }
            json.dump(comparison, f, indent=2)

if __name__ == '__main__':
    main()
