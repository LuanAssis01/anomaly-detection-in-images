"""
Script de treinamento para todos os modelos
"""
import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from tqdm import tqdm
import argparse
from datetime import datetime

# Adicionar o diretório raiz ao path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from configs.config import *
from utils.dataset import ForgeryDataset, get_transforms, custom_collate_fn
from utils.metrics import calculate_metrics, AverageMeter
from models import CNNModel, ViTModel, CvTModel, DINOv2Model

def set_seed(seed=42):
    """Define seed para reprodutibilidade"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Treina por uma época"""
    model.train()
    
    losses = AverageMeter()
    accuracies = AverageMeter()
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} - Train')
    
    for images, labels, _ in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Métricas
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == labels).float().mean().item() * 100
        
        losses.update(loss.item(), images.size(0))
        accuracies.update(accuracy, images.size(0))
        
        pbar.set_postfix({
            'loss': f'{losses.avg:.4f}',
            'acc': f'{accuracies.avg:.2f}%'
        })
    
    return losses.avg, accuracies.avg

def validate_epoch(model, dataloader, criterion, device):
    """Valida o modelo"""
    model.eval()
    
    losses = AverageMeter()
    
    all_labels = []
    all_predictions = []
    all_probs = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        
        for images, labels, _ in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Métricas
            probs = torch.softmax(outputs, dim=1)[:, 1]  # Prob da classe "forged"
            _, predicted = torch.max(outputs.data, 1)
            
            losses.update(loss.item(), images.size(0))
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{losses.avg:.4f}'})
    
    # Calcular métricas
    metrics = calculate_metrics(
        np.array(all_labels),
        np.array(all_predictions),
        np.array(all_probs)
    )
    
    return losses.avg, metrics

def train_model(model_type: str, num_epochs: int = NUM_EPOCHS, batch_size: int = BATCH_SIZE):
    """
    Treina um modelo específico
    
    Args:
        model_type: 'cnn', 'vit', 'cvt', 'dinov2'
        num_epochs: Número de épocas
        batch_size: Tamanho do batch
    """
    print(f"\n{'='*60}")
    print(f"Treinando modelo: {MODEL_CONFIGS[model_type]['name']}")
    print(f"{'='*60}\n")
    
    # Configurar device
    device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Criar diretórios
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    
    # Dataset e DataLoader
    print("\nCarregando dataset...")
    train_dataset = ForgeryDataset(
        root_dir=TRAIN_DIR,
        masks_dir=TRAIN_MASKS_DIR,
        transform=get_transforms(IMAGE_SIZE, mode='train'),
        mode='train'
    )
    
    # Split train/val (80/20)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(
        train_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Criar modelo
    print(f"\nCriando modelo {model_type}...")
    if model_type in ['resnet50', 'resnet101']:
        model = CNNModel(
            num_classes=NUM_CLASSES,
            model_name=MODEL_CONFIGS[model_type]['model_name'],
            pretrained=True
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
    
    model = model.to(device)
    print(f"Parâmetros treináveis: {model.count_parameters():,}")
    
    # Loss e optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Histórico de treinamento
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'val_metrics': []
    }
    
    best_val_acc = 0.0
    best_epoch = 0
    
    # Treinamento
    print("\nIniciando treinamento...\n")
    for epoch in range(1, num_epochs + 1):
        # Treinar
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validar
        val_loss, val_metrics = validate_epoch(
            model, val_loader, criterion, device
        )
        
        val_acc = val_metrics['accuracy'] * 100
        
        # Atualizar scheduler
        scheduler.step(val_loss)
        
        # Salvar histórico
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['val_metrics'].append(val_metrics)
        
        # Print resumo da época
        print(f"\nEpoch {epoch}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"  Precision: {val_metrics['precision']:.4f} | Recall: {val_metrics['recall']:.4f}")
        print(f"  F1-Score: {val_metrics['f1_score']:.4f}")
        
        # Salvar melhor modelo
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            
            checkpoint_path = os.path.join(CHECKPOINTS_DIR, f'{model_type}_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_metrics': val_metrics
            }, checkpoint_path)
            
            print(f"  ✓ Modelo salvo! (melhor acc: {best_val_acc:.2f}%)")
    
    print(f"\n{'='*60}")
    print(f"Treinamento concluído!")
    print(f"Melhor acurácia: {best_val_acc:.2f}% (Época {best_epoch})")
    print(f"{'='*60}\n")
    
    # Salvar histórico
    history_path = os.path.join(RESULTS_DIR, f'{model_type}_history.json')
    with open(history_path, 'w') as f:
        # Converter numpy arrays para listas
        history_serializable = {
            k: [float(x) if isinstance(x, (np.floating, float)) else x for x in v]
            for k, v in history.items() if k != 'val_metrics'
        }
        json.dump(history_serializable, f, indent=2)
    
    return model, history

def main():
    parser = argparse.ArgumentParser(description='Treinar modelos de detecção de falsificação')
    parser.add_argument('--model', type=str, default='all',
                      choices=['all', 'resnet50', 'resnet101', 'vit', 'cvt13', 'cvt21', 'cvt_w24', 'dinov2'],
                      help='Modelo a treinar')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS,
                      help='Número de épocas')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                      help='Tamanho do batch')
    
    args = parser.parse_args()
    
    # Definir seed
    set_seed(RANDOM_SEED)
    
    # Treinar modelos
    if args.model == 'all':
        models_to_train = ['resnet50', 'vit', 'cvt13', 'cvt21', 'cvt_w24', 'dinov2']
    else:
        models_to_train = [args.model]
    
    for model_type in models_to_train:
        train_model(model_type, args.epochs, args.batch_size)

if __name__ == '__main__':
    main()
