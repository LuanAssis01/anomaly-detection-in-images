"""
Script de treinamento para todos os modelos
"""
import os
import sys
import math
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
import numpy as np
from tqdm import tqdm
import argparse
from datetime import datetime

# Adicionar o diretório raiz ao path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from configs.config import *
# Importar novos parâmetros com defaults seguros caso o config antigo não tenha
try:
    from configs.config import USE_AMP, GRADIENT_ACCUMULATION_STEPS, USE_COMPILE
except ImportError:
    USE_AMP = True
    GRADIENT_ACCUMULATION_STEPS = 1
    USE_COMPILE = True
try:
    from configs.config import FINETUNE_CONFIGS
except ImportError:
    FINETUNE_CONFIGS = {}
try:
    from configs.config import CLASS_WEIGHTS
except ImportError:
    CLASS_WEIGHTS = None
from utils.dataset import ForgeryDataset, get_transforms, custom_collate_fn, stratified_split
from utils.metrics import calculate_metrics, AverageMeter
from models import CNNModel, ViTModel, CvTModel, DINOv2Model

def set_seed(seed=42):
    """Define seed para reprodutibilidade"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True  # Seguro com input fixo (224x224), ~10-20% mais rápido

def train_epoch(model, dataloader, criterion, optimizer, device, epoch, scaler=None, accumulation_steps=1):
    """Treina por uma época com suporte a AMP e gradient accumulation"""
    model.train()
    
    losses = AverageMeter()
    accuracies = AverageMeter()
    use_amp = scaler is not None
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} - Train')
    
    for step, (images, labels, _) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # Forward pass com AMP (mixed precision)
        with torch.amp.autocast('cuda', enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss = loss / accumulation_steps  # Normalizar para gradient accumulation
        
        # Backward pass com GradScaler
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Atualizar pesos a cada N steps (gradient accumulation)
        if (step + 1) % accumulation_steps == 0 or (step + 1) == len(dataloader):
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)  # Mais rápido e usa menos memória
        
        # Métricas
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == labels).float().mean().item() * 100
        
        losses.update(loss.item() * accumulation_steps, images.size(0))  # Desnormalizar para log
        accuracies.update(accuracy, images.size(0))
        
        pbar.set_postfix({
            'loss': f'{losses.avg:.4f}',
            'acc': f'{accuracies.avg:.2f}%'
        })
    
    return losses.avg, accuracies.avg

def validate_epoch(model, dataloader, criterion, device, use_amp=False):
    """Valida o modelo com suporte a AMP"""
    model.eval()
    
    losses = AverageMeter()
    
    all_labels = []
    all_predictions = []
    all_probs = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        
        for images, labels, _ in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Forward pass com AMP
            with torch.amp.autocast('cuda', enabled=use_amp):
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

def train_model(model_type: str, num_epochs: int = NUM_EPOCHS, batch_size: int = BATCH_SIZE, accumulation_steps: int = GRADIENT_ACCUMULATION_STEPS):
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
    
    # Configurar device (GPU obrigatória)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA não disponível. Este projeto requer GPU para execução.")
    device = torch.device('cuda')
    print(f"Device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Criar diretórios
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    
    # Dataset e DataLoader
    # Usar image_size específico do modelo (ex: CvT-W24 usa 384) ou o padrão global
    model_image_size = MODEL_CONFIGS[model_type].get('image_size', IMAGE_SIZE)
    print(f"Image size: {model_image_size}x{model_image_size}")
    
    print("\nCarregando dataset...")
    full_dataset = ForgeryDataset(
        root_dir=TRAIN_DIR,
        masks_dir=TRAIN_MASKS_DIR,
        transform=get_transforms(model_image_size, mode='train'),
        mode='train'
    )

    # Split estratificado train/val (80/20) — garante distribuição igual de classes
    train_idx, val_idx = stratified_split(full_dataset, val_ratio=0.2, seed=RANDOM_SEED)
    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True if NUM_WORKERS > 0 else False,
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
        collate_fn=custom_collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True if NUM_WORKERS > 0 else False,
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
        collate_fn=custom_collate_fn
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Criar modelo
    print(f"\nCriando modelo {model_type}...")
    if model_type == 'resnet50':
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
    elif model_type in ['cvt13', 'cvt21']:
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
    
    # torch.compile() — funde operações para execução mais rápida na GPU
    if USE_COMPILE and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model)
            print("✓ torch.compile() ativado")
        except Exception as e:
            print(f"⚠ torch.compile() não disponível: {e}")
    
    # AMP — Mixed Precision (FP16) com GradScaler
    use_amp = USE_AMP
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    if use_amp:
        print("✓ AMP (Mixed Precision FP16) ativado")
    if accumulation_steps > 1:
        print(f"✓ Gradient Accumulation: {accumulation_steps} steps (batch efetivo = {batch_size * accumulation_steps})")
    
    # Loss e optimizer (com pesos de classe para penalizar FN)
    if CLASS_WEIGHTS is not None:
        weight = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=weight)
        print(f"✓ Class weights: {CLASS_WEIGHTS}")
    else:
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
            model, train_loader, criterion, optimizer, device, epoch,
            scaler=scaler, accumulation_steps=accumulation_steps
        )
        
        # Validar
        val_loss, val_metrics = validate_epoch(
            model, val_loader, criterion, device, use_amp=use_amp
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


def train_model_finetuned(model_type: str, batch_size: int = BATCH_SIZE):
    """
    Treina um modelo com fine-tuning em 2 fases:
      Fase 1: backbone congelado, treina só o classificador
      Fase 2: backbone descongelado, fine-tuning com lr diferenciado + cosine scheduler
    """
    ft_cfg = FINETUNE_CONFIGS[model_type]
    accumulation_steps = ft_cfg.get('accumulation_steps', 1)
    total_epochs = ft_cfg['phase1_epochs'] + ft_cfg['phase2_epochs']
    effective_batch = batch_size * accumulation_steps
    
    print(f"\n{'='*60}")
    print(f"Fine-tuning modelo: {MODEL_CONFIGS[model_type]['name']}")
    print(f"  Batch size: {batch_size} (efetivo: {effective_batch} com accumulation={accumulation_steps})")
    print(f"  Fase 1: {ft_cfg['phase1_epochs']} épocas (backbone congelado, lr={ft_cfg['phase1_lr']})")
    print(f"  Fase 2: {ft_cfg['phase2_epochs']} épocas (fine-tuning completo, backbone_lr={ft_cfg['phase2_backbone_lr']}, classifier_lr={ft_cfg['phase2_classifier_lr']})")
    print(f"{'='*60}\n")
    
    # Configurar device (GPU obrigatória)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA não disponível. Este projeto requer GPU para execução.")
    device = torch.device('cuda')
    print(f"Device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Criar diretórios
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    
    # Dataset e DataLoader
    model_image_size = MODEL_CONFIGS[model_type].get('image_size', IMAGE_SIZE)
    print(f"Image size: {model_image_size}x{model_image_size}")
    
    print("\nCarregando dataset...")
    full_dataset = ForgeryDataset(
        root_dir=TRAIN_DIR,
        masks_dir=TRAIN_MASKS_DIR,
        transform=get_transforms(model_image_size, mode='train'),
        mode='train'
    )

    # Split estratificado train/val (80/20) — garante distribuição igual de classes
    train_idx, val_idx = stratified_split(full_dataset, val_ratio=0.2, seed=RANDOM_SEED)
    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=True if NUM_WORKERS > 0 else False,
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
        collate_fn=custom_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=True if NUM_WORKERS > 0 else False,
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
        collate_fn=custom_collate_fn
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Criar modelo
    print(f"\nCriando modelo {model_type}...")
    if model_type == 'vit':
        model = ViTModel(
            model_name=MODEL_CONFIGS['vit']['model_name'],
            num_classes=NUM_CLASSES
        )
    elif model_type in ['cvt13', 'cvt21']:
        model = CvTModel(
            model_name=MODEL_CONFIGS[model_type]['model_name'],
            num_classes=NUM_CLASSES
        )
    elif model_type == 'dinov2':
        model = DINOv2Model(
            model_name=MODEL_CONFIGS['dinov2']['model_name'],
            num_classes=NUM_CLASSES
        )
    elif model_type == 'resnet50':
        model = CNNModel(
            num_classes=NUM_CLASSES,
            model_name=MODEL_CONFIGS[model_type]['model_name'],
            pretrained=True
        )
    else:
        raise ValueError(f"Modelo {model_type} não tem configuração de fine-tuning")
    
    model = model.to(device)
    print(f"Parâmetros totais: {model.count_parameters():,}")
    
    # AMP
    use_amp = USE_AMP
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    if use_amp:
        print("✓ AMP (Mixed Precision FP16) ativado")

    label_smoothing = ft_cfg.get('label_smoothing', 0.0)
    if CLASS_WEIGHTS is not None:
        weight = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)
        print(f"✓ Class weights: {CLASS_WEIGHTS}")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    if label_smoothing > 0:
        print(f"✓ Label Smoothing: {label_smoothing}")

    early_stopping_patience = ft_cfg.get('early_stopping_patience', 999)

    # Histórico de treinamento
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'val_metrics': []
    }
    
    best_val_acc = 0.0
    best_epoch = 0
    
    # ========== FASE 1: Backbone congelado ==========
    print(f"\n{'─'*60}")
    print(f"FASE 1: Treinando apenas o classificador ({ft_cfg['phase1_epochs']} épocas)")
    print(f"{'─'*60}")
    
    model.freeze_backbone()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parâmetros treináveis (fase 1): {trainable:,}")
    
    optimizer_p1 = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=ft_cfg['phase1_lr'],
        weight_decay=ft_cfg['weight_decay']
    )
    scheduler_p1 = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_p1, T_max=ft_cfg['phase1_epochs']
    )
    
    # torch.compile() após congelar
    compiled_model = model
    if USE_COMPILE and hasattr(torch, 'compile'):
        try:
            compiled_model = torch.compile(model)
            print("✓ torch.compile() ativado")
        except Exception as e:
            print(f"⚠ torch.compile() não disponível: {e}")
            compiled_model = model
    
    for epoch in range(1, ft_cfg['phase1_epochs'] + 1):
        train_loss, train_acc = train_epoch(
            compiled_model, train_loader, criterion, optimizer_p1, device, epoch,
            scaler=scaler, accumulation_steps=accumulation_steps
        )
        val_loss, val_metrics = validate_epoch(
            compiled_model, val_loader, criterion, device, use_amp=use_amp
        )
        val_acc = val_metrics['accuracy'] * 100
        
        scheduler_p1.step()
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['val_metrics'].append(val_metrics)
        
        print(f"\n[Fase 1] Epoch {epoch}/{ft_cfg['phase1_epochs']}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"  F1: {val_metrics['f1_score']:.4f} | LR: {optimizer_p1.param_groups[0]['lr']:.2e}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            checkpoint_path = os.path.join(CHECKPOINTS_DIR, f'{model_type}_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'val_metrics': val_metrics,
                'phase': 1
            }, checkpoint_path)
            print(f"  ✓ Modelo salvo! (melhor acc: {best_val_acc:.2f}%)")
    
    print(f"\nFase 1 concluída! Melhor acc: {best_val_acc:.2f}%")
    
    # ========== FASE 2: Fine-tuning completo ==========
    print(f"\n{'─'*60}")
    print(f"FASE 2: Fine-tuning completo ({ft_cfg['phase2_epochs']} épocas)")
    print(f"{'─'*60}")

    no_improve_count = 0  # contador para early stopping

    model.unfreeze_backbone()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parâmetros treináveis (fase 2): {trainable:,}")
    
    # Recompilar o modelo após descongelar
    compiled_model = model
    if USE_COMPILE and hasattr(torch, 'compile'):
        try:
            compiled_model = torch.compile(model)
            print("✓ torch.compile() reativado")
        except Exception as e:
            print(f"⚠ torch.compile() não disponível: {e}")
            compiled_model = model
    
    # Otimizador com learning rates diferenciados
    param_groups = model.get_param_groups(
        backbone_lr=ft_cfg['phase2_backbone_lr'],
        classifier_lr=ft_cfg['phase2_classifier_lr']
    )
    optimizer_p2 = optim.AdamW(
        param_groups,
        weight_decay=ft_cfg['weight_decay']
    )
    
    # Cosine Annealing scheduler com warm-up linear
    warmup_epochs = ft_cfg['warmup_epochs']
    phase2_epochs = ft_cfg['phase2_epochs']
    
    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            # Warm-up linear: de 0 até 1
            return float(current_epoch + 1) / float(warmup_epochs)
        else:
            # Cosine annealing de 1 até 0
            progress = float(current_epoch - warmup_epochs) / float(max(1, phase2_epochs - warmup_epochs))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    scheduler_p2 = optim.lr_scheduler.LambdaLR(optimizer_p2, lr_lambda)
    
    for epoch in range(1, ft_cfg['phase2_epochs'] + 1):
        global_epoch = ft_cfg['phase1_epochs'] + epoch
        
        train_loss, train_acc = train_epoch(
            compiled_model, train_loader, criterion, optimizer_p2, device, global_epoch,
            scaler=scaler, accumulation_steps=accumulation_steps
        )
        val_loss, val_metrics = validate_epoch(
            compiled_model, val_loader, criterion, device, use_amp=use_amp
        )
        val_acc = val_metrics['accuracy'] * 100
        
        scheduler_p2.step()
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['val_metrics'].append(val_metrics)
        
        backbone_lr = optimizer_p2.param_groups[0]['lr']
        classifier_lr = optimizer_p2.param_groups[1]['lr']
        
        print(f"\n[Fase 2] Epoch {epoch}/{ft_cfg['phase2_epochs']} (global: {global_epoch})")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"  F1: {val_metrics['f1_score']:.4f} | Backbone LR: {backbone_lr:.2e} | Classifier LR: {classifier_lr:.2e}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = global_epoch
            no_improve_count = 0
            checkpoint_path = os.path.join(CHECKPOINTS_DIR, f'{model_type}_best.pth')
            torch.save({
                'epoch': global_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer_p2.state_dict(),
                'val_acc': val_acc,
                'val_metrics': val_metrics,
                'phase': 2
            }, checkpoint_path)
            print(f"  ✓ Modelo salvo! (melhor acc: {best_val_acc:.2f}%)")
        else:
            no_improve_count += 1
            print(f"  ⏱  Sem melhora: {no_improve_count}/{early_stopping_patience}")
            if no_improve_count >= early_stopping_patience:
                print(f"\n⚠ Early stopping ativado na época {global_epoch} (paciência={early_stopping_patience})")
                break
    
    print(f"\n{'='*60}")
    print(f"Fine-tuning concluído!")
    print(f"Melhor acurácia: {best_val_acc:.2f}% (Época {best_epoch})")
    print(f"{'='*60}\n")
    
    # Salvar histórico
    history_path = os.path.join(RESULTS_DIR, f'{model_type}_history.json')
    with open(history_path, 'w') as f:
        history_serializable = {
            k: [float(x) if isinstance(x, (np.floating, float)) else x for x in v]
            for k, v in history.items() if k != 'val_metrics'
        }
        json.dump(history_serializable, f, indent=2)
    
    return model, history


def main():
    parser = argparse.ArgumentParser(description='Treinar modelos de detecção de falsificação')
    parser.add_argument('--model', type=str, default='all',
                      choices=['all', 'resnet50', 'vit', 'cvt13', 'cvt21', 'dinov2'],
                      help='Modelo a treinar')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS,
                      help='Número de épocas')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                      help='Tamanho do batch')
    parser.add_argument('--accumulation_steps', type=int, default=GRADIENT_ACCUMULATION_STEPS,
                      help='Steps de acumulação de gradiente (simula batch maior)')
    
    args = parser.parse_args()
    
    # Definir seed
    set_seed(RANDOM_SEED)
    
    # Treinar modelos
    if args.model == 'all':
        models_to_train = ['resnet50', 'vit', 'cvt13', 'cvt21', 'dinov2']
    else:
        models_to_train = [args.model]
    
    for model_type in models_to_train:
        # Usar batch_size específico do modelo se não foi passado via CLI
        model_batch_size = MODEL_CONFIGS[model_type].get('batch_size', args.batch_size)
        
        # Usar fine-tuning específico se configurado, senão treinamento padrão
        if model_type in FINETUNE_CONFIGS:
            print(f"\n>>> Usando fine-tuning específico para {model_type}")
            train_model_finetuned(model_type, model_batch_size)
        else:
            train_model(model_type, args.epochs, model_batch_size, args.accumulation_steps)

if __name__ == '__main__':
    main()
