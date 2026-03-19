"""
Script de treinamento para ResNet-50 e DINOv2
Suporta cenários com e sem dados sintéticos, com medição de tempo.
"""
import os
import sys
import math
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm
import argparse
import subprocess

# Adicionar o diretório raiz ao path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from configs.config import *
from configs.config import (
    USE_AMP, GRADIENT_ACCUMULATION_STEPS, USE_COMPILE,
    FINETUNE_CONFIGS, CLASS_WEIGHTS, SCENARIOS
)
from utils.dataset import ForgeryDataset, get_transforms, custom_collate_fn, stratified_split
from utils.metrics import calculate_metrics, AverageMeter
from models import CNNModel, DINOv2Model


def format_time(seconds):
    """Formata segundos em HH:MM:SS"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


def set_seed(seed=42):
    """Define seed para reprodutibilidade"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


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

        with torch.amp.autocast('cuda', enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss = loss / accumulation_steps

        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % accumulation_steps == 0 or (step + 1) == len(dataloader):
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == labels).float().mean().item() * 100

        losses.update(loss.item() * accumulation_steps, images.size(0))
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

            with torch.amp.autocast('cuda', enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)

            probs = torch.softmax(outputs, dim=1)[:, 1]
            _, predicted = torch.max(outputs.data, 1)

            losses.update(loss.item(), images.size(0))

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            pbar.set_postfix({'loss': f'{losses.avg:.4f}'})

    metrics = calculate_metrics(
        np.array(all_labels),
        np.array(all_predictions),
        np.array(all_probs)
    )

    return losses.avg, metrics


def create_model(model_type, device):
    """Cria modelo ResNet-50 ou DINOv2"""
    if model_type == 'resnet50':
        model = CNNModel(
            num_classes=NUM_CLASSES,
            model_name=MODEL_CONFIGS[model_type]['model_name'],
            pretrained=True
        )
    elif model_type == 'dinov2':
        model = DINOv2Model(
            model_name=MODEL_CONFIGS['dinov2']['model_name'],
            num_classes=NUM_CLASSES
        )
    else:
        raise ValueError(f"Modelo desconhecido: {model_type}")
    return model.to(device)


def create_dataloaders(scenario, model_type, batch_size):
    """Cria DataLoaders a partir do cenário"""
    scenario_cfg = SCENARIOS[scenario]
    train_dir = scenario_cfg['train_dir']
    masks_dir = scenario_cfg['masks_dir']

    model_image_size = MODEL_CONFIGS[model_type].get('image_size', IMAGE_SIZE)

    full_dataset = ForgeryDataset(
        root_dir=train_dir,
        masks_dir=masks_dir,
        transform=get_transforms(model_image_size, mode='train'),
        mode='train'
    )

    train_idx, val_idx = stratified_split(full_dataset, val_ratio=0.2, seed=RANDOM_SEED)
    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True if NUM_WORKERS > 0 else False,
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
        collate_fn=custom_collate_fn
    )

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, len(train_dataset), len(val_dataset)


def train_model_finetuned(model_type: str, scenario: str, batch_size: int = BATCH_SIZE):
    """
    Treina um modelo com fine-tuning em 2 fases com medição de tempo.

    Args:
        model_type: 'resnet50' ou 'dinov2'
        scenario: 'no_synthetic' ou 'with_synthetic'

    Returns:
        dict com histórico e tempos
    """
    ft_cfg = FINETUNE_CONFIGS[model_type]
    accumulation_steps = ft_cfg.get('accumulation_steps', 1)
    effective_batch = batch_size * accumulation_steps
    run_name = f"{model_type}_{scenario}"

    print(f"\n{'='*60}")
    print(f"Fine-tuning: {MODEL_CONFIGS[model_type]['name']}  |  Cenário: {scenario}")
    print(f"  Batch size: {batch_size} (efetivo: {effective_batch})")
    print(f"  Fase 1: {ft_cfg['phase1_epochs']} épocas (backbone congelado)")
    print(f"  Fase 2: {ft_cfg['phase2_epochs']} épocas (fine-tuning completo)")
    print(f"{'='*60}\n")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA não disponível. Este projeto requer GPU para execução.")
    device = torch.device('cuda')
    print(f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB)")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

    # DataLoaders
    print(f"\nCarregando dataset ({scenario})...")
    train_loader, val_loader, n_train, n_val = create_dataloaders(scenario, model_type, batch_size)
    print(f"Train: {n_train} | Val: {n_val}")

    # Modelo
    print(f"\nCriando modelo {model_type}...")
    model = create_model(model_type, device)
    print(f"Parâmetros totais: {model.count_parameters():,}")

    # AMP
    use_amp = USE_AMP
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    if use_amp:
        print("AMP (FP16) ativado")

    label_smoothing = ft_cfg.get('label_smoothing', 0.0)
    if CLASS_WEIGHTS is not None:
        weight = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    early_stopping_patience = ft_cfg.get('early_stopping_patience', 999)

    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'val_metrics': []
    }

    best_val_acc = 0.0
    best_epoch = 0

    # ==================== INÍCIO DA MEDIÇÃO DE TEMPO ====================
    train_start = time.time()

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

    compiled_model = model
    if USE_COMPILE and hasattr(torch, 'compile'):
        try:
            compiled_model = torch.compile(model)
            print("torch.compile() ativado")
        except Exception:
            compiled_model = model

    phase1_start = time.time()
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
            checkpoint_path = os.path.join(CHECKPOINTS_DIR, f'{run_name}_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'val_metrics': val_metrics,
                'scenario': scenario,
                'phase': 1
            }, checkpoint_path)
            print(f"  Modelo salvo! (melhor acc: {best_val_acc:.2f}%)")

    phase1_time = time.time() - phase1_start
    print(f"\nFase 1 concluída em {format_time(phase1_time)}. Melhor acc: {best_val_acc:.2f}%")

    # ========== FASE 2: Fine-tuning completo ==========
    print(f"\n{'─'*60}")
    print(f"FASE 2: Fine-tuning completo ({ft_cfg['phase2_epochs']} épocas)")
    print(f"{'─'*60}")

    no_improve_count = 0

    model.unfreeze_backbone()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parâmetros treináveis (fase 2): {trainable:,}")

    compiled_model = model
    if USE_COMPILE and hasattr(torch, 'compile'):
        try:
            compiled_model = torch.compile(model)
        except Exception:
            compiled_model = model

    param_groups = model.get_param_groups(
        backbone_lr=ft_cfg['phase2_backbone_lr'],
        classifier_lr=ft_cfg['phase2_classifier_lr']
    )
    optimizer_p2 = optim.AdamW(
        param_groups,
        weight_decay=ft_cfg['weight_decay']
    )

    warmup_epochs = ft_cfg['warmup_epochs']
    phase2_epochs = ft_cfg['phase2_epochs']

    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            return float(current_epoch + 1) / float(warmup_epochs)
        else:
            progress = float(current_epoch - warmup_epochs) / float(max(1, phase2_epochs - warmup_epochs))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler_p2 = optim.lr_scheduler.LambdaLR(optimizer_p2, lr_lambda)

    phase2_start = time.time()
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
            checkpoint_path = os.path.join(CHECKPOINTS_DIR, f'{run_name}_best.pth')
            torch.save({
                'epoch': global_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer_p2.state_dict(),
                'val_acc': val_acc,
                'val_metrics': val_metrics,
                'scenario': scenario,
                'phase': 2
            }, checkpoint_path)
            print(f"  Modelo salvo! (melhor acc: {best_val_acc:.2f}%)")
        else:
            no_improve_count += 1
            print(f"  Sem melhora: {no_improve_count}/{early_stopping_patience}")
            if no_improve_count >= early_stopping_patience:
                print(f"\nEarly stopping na época {global_epoch}")
                break

    phase2_time = time.time() - phase2_start
    total_train_time = time.time() - train_start

    print(f"\n{'='*60}")
    print(f"Fine-tuning concluído!")
    print(f"Melhor acurácia: {best_val_acc:.2f}% (Época {best_epoch})")
    print(f"Tempo fase 1: {format_time(phase1_time)}")
    print(f"Tempo fase 2: {format_time(phase2_time)}")
    print(f"Tempo total:  {format_time(total_train_time)}")
    print(f"{'='*60}\n")

    # Salvar histórico com tempos
    history_data = {
        'train_loss': [float(x) if isinstance(x, (np.floating, float)) else x for x in history['train_loss']],
        'val_loss': [float(x) if isinstance(x, (np.floating, float)) else x for x in history['val_loss']],
        'train_acc': [float(x) if isinstance(x, (np.floating, float)) else x for x in history['train_acc']],
        'val_acc': [float(x) if isinstance(x, (np.floating, float)) else x for x in history['val_acc']],
        'timing': {
            'phase1_seconds': round(phase1_time, 2),
            'phase2_seconds': round(phase2_time, 2),
            'total_train_seconds': round(total_train_time, 2),
            'phase1_formatted': format_time(phase1_time),
            'phase2_formatted': format_time(phase2_time),
            'total_train_formatted': format_time(total_train_time),
        },
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch,
        'scenario': scenario,
        'model_type': model_type,
    }

    history_path = os.path.join(RESULTS_DIR, f'{run_name}_history.json')
    with open(history_path, 'w') as f:
        json.dump(history_data, f, indent=2)

    return {
        'model_type': model_type,
        'scenario': scenario,
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch,
        'phase1_time': phase1_time,
        'phase2_time': phase2_time,
        'total_train_time': total_train_time,
    }


def generate_data_for_scenario(scenario, target=None):
    """Chama generate_data.py para um cenário específico"""
    cmd = [sys.executable, os.path.join(BASE_DIR, 'src', 'generate_data.py'), '--scenario', scenario]
    if target:
        cmd.extend(['--target', str(target)])
    print(f"\n{'='*60}")
    print(f"Gerando dados para cenário: {scenario}")
    print(f"{'='*60}")
    subprocess.run(cmd, check=True)


def print_summary(results):
    """Imprime tabela comparativa dos resultados"""
    if not results:
        return

    print(f"\n{'='*80}")
    print(f"RESUMO COMPARATIVO DE TREINAMENTO")
    print(f"{'='*80}")
    print(f"{'Modelo':<12} {'Cenário':<18} {'Acc':<10} {'Época':<8} {'Fase 1':<12} {'Fase 2':<12} {'Total':<12}")
    print(f"{'─'*80}")

    for r in results:
        print(f"{r['model_type']:<12} "
              f"{r['scenario']:<18} "
              f"{r['best_val_acc']:<10.2f} "
              f"{r['best_epoch']:<8} "
              f"{format_time(r['phase1_time']):<12} "
              f"{format_time(r['phase2_time']):<12} "
              f"{format_time(r['total_train_time']):<12}")

    print(f"{'='*80}\n")

    # Salvar resumo em JSON
    summary_path = os.path.join(RESULTS_DIR, 'training_summary.json')
    summary = []
    for r in results:
        summary.append({
            'model': r['model_type'],
            'scenario': r['scenario'],
            'best_val_acc': round(r['best_val_acc'], 2),
            'best_epoch': r['best_epoch'],
            'phase1_time': format_time(r['phase1_time']),
            'phase2_time': format_time(r['phase2_time']),
            'total_train_time': format_time(r['total_train_time']),
            'total_seconds': round(r['total_train_time'], 2),
        })
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Resumo salvo em: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description='Treinar ResNet-50 e DINOv2 para detecção de falsificação')
    parser.add_argument('--model', type=str, default='all',
                      choices=['all', 'resnet50', 'dinov2'],
                      help='Modelo a treinar (default: all = resnet50 + dinov2)')
    parser.add_argument('--scenario', type=str, default='all',
                      choices=['all', 'no_augmentation', 'no_synthetic', 'with_synthetic'],
                      help='Cenário de dados (default: all = todos os 3 cenários)')
    parser.add_argument('--generate-data', action='store_true',
                      help='Gerar dados antes de treinar (chama generate_data.py)')
    parser.add_argument('--target', type=int, default=None,
                      help='Número alvo de imagens por classe (usado com --generate-data)')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                      help='Tamanho do batch')

    args = parser.parse_args()

    set_seed(RANDOM_SEED)

    models_to_train = ['resnet50', 'dinov2'] if args.model == 'all' else [args.model]
    scenarios = ['no_augmentation', 'no_synthetic', 'with_synthetic'] if args.scenario == 'all' else [args.scenario]

    print(f"\nModelos:   {models_to_train}")
    print(f"Cenários:  {scenarios}")

    # Gerar dados se solicitado
    if args.generate_data:
        for scenario in scenarios:
            generate_data_for_scenario(scenario, args.target)

    # Treinar cada combinação modelo x cenário
    all_results = []
    total_start = time.time()

    for scenario in scenarios:
        for model_type in models_to_train:
            model_batch_size = MODEL_CONFIGS[model_type].get('batch_size', args.batch_size)

            if model_type in FINETUNE_CONFIGS:
                result = train_model_finetuned(model_type, scenario, model_batch_size)
            else:
                raise ValueError(f"Modelo {model_type} sem configuração de fine-tuning")

            all_results.append(result)

    total_elapsed = time.time() - total_start

    # Resumo final
    print_summary(all_results)
    print(f"Tempo total de todas as execuções: {format_time(total_elapsed)}")


if __name__ == '__main__':
    main()
