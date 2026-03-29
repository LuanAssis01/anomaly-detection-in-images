"""
Hyperparameter Search para otimização de modelos de detecção de anomalias.

Implementa Grid Search e Randomized Search com épocas reduzidas para
avaliação rápida, seguido de retreino com a melhor configuração encontrada.

Uso:
    # Grid Search no ResNet-50
    python src/hyperparam_search.py --model resnet50 --scenario no_augmentation --method grid

    # Randomized Search no DINOv2 (20 combinações)
    python src/hyperparam_search.py --model dinov2 --scenario with_synthetic --method random --n-iter 20

    # Grid Search em todos os modelos e cenários
    python src/hyperparam_search.py --model all --scenario all --method grid
"""
import os
import sys
import json
import time
import math
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import argparse

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from configs.config import (
    NUM_CLASSES, IMAGE_SIZE, BATCH_SIZE, RANDOM_SEED,
    USE_AMP, GRADIENT_ACCUMULATION_STEPS, NUM_WORKERS,
    MODEL_CONFIGS, SCENARIOS, RESULTS_DIR, CHECKPOINTS_DIR,
    VAL_RATIO, TEST_RATIO
)
from utils.dataset import (
    ForgeryDataset, get_transforms, custom_collate_fn,
    TransformSubset, stratified_train_val_test_split, save_split
)
from utils.metrics import calculate_metrics, AverageMeter
from models import CNNModel, DINOv2Model


# ============================================================================
# Espaços de busca por modelo
# ============================================================================

SEARCH_SPACES = {
    'resnet50': {
        'grid': {
            'phase1_lr': [3e-4, 5e-4, 1e-3],
            'phase2_backbone_lr': [1e-5, 2e-5, 5e-5],
            'phase2_classifier_lr': [1e-4, 2e-4, 5e-4],
            'weight_decay': [1e-4, 5e-4, 1e-3],
            'label_smoothing': [0.05, 0.1],
            'class_weights_1': [1.0, 1.3, 1.5],
        },
        'random': {
            'phase1_lr': ('log_uniform', 1e-4, 2e-3),
            'phase2_backbone_lr': ('log_uniform', 5e-6, 1e-4),
            'phase2_classifier_lr': ('log_uniform', 5e-5, 1e-3),
            'weight_decay': ('log_uniform', 1e-5, 1e-2),
            'label_smoothing': ('uniform', 0.0, 0.15),
            'class_weights_1': ('uniform', 0.8, 2.0),
        },
    },
    'dinov2': {
        'grid': {
            'phase1_lr': [3e-4, 5e-4, 8e-4],
            'phase2_backbone_lr': [2e-6, 5e-6, 1e-5],
            'phase2_classifier_lr': [3e-5, 5e-5, 1e-4],
            'weight_decay': [0.01, 0.05, 0.1],
            'label_smoothing': [0.05, 0.1],
            'class_weights_1': [1.0, 1.3, 1.5],
        },
        'random': {
            'phase1_lr': ('log_uniform', 1e-4, 2e-3),
            'phase2_backbone_lr': ('log_uniform', 1e-6, 5e-5),
            'phase2_classifier_lr': ('log_uniform', 1e-5, 5e-4),
            'weight_decay': ('log_uniform', 1e-3, 0.2),
            'label_smoothing': ('uniform', 0.0, 0.15),
            'class_weights_1': ('uniform', 0.8, 2.0),
        },
    },
}

# Épocas reduzidas para busca rápida
SEARCH_EPOCHS = {
    'resnet50': {'phase1': 3, 'phase2': 8},
    'dinov2': {'phase1': 4, 'phase2': 8},
}


# ============================================================================
# Funções auxiliares
# ============================================================================

def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def sample_random_param(spec):
    """Amostra um valor de um espaço de busca aleatório."""
    dist_type = spec[0]
    if dist_type == 'log_uniform':
        low, high = np.log(spec[1]), np.log(spec[2])
        return float(np.exp(np.random.uniform(low, high)))
    elif dist_type == 'uniform':
        return float(np.random.uniform(spec[1], spec[2]))
    elif dist_type == 'choice':
        return np.random.choice(spec[1])
    else:
        raise ValueError(f"Distribuição desconhecida: {dist_type}")


def generate_grid_configs(search_space):
    """Gera todas as combinações do Grid Search."""
    keys = list(search_space.keys())
    values = list(search_space.values())
    configs = []
    for combo in itertools.product(*values):
        config = dict(zip(keys, combo))
        configs.append(config)
    return configs


def generate_random_configs(search_space, n_iter):
    """Gera n_iter combinações aleatórias para Randomized Search."""
    configs = []
    for _ in range(n_iter):
        config = {}
        for key, spec in search_space.items():
            config[key] = sample_random_param(spec)
        configs.append(config)
    return configs


def create_model(model_type, device):
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
    """Cria DataLoaders com política assimétrica de transforms."""
    scenario_cfg = SCENARIOS[scenario]
    train_dir = scenario_cfg['train_dir']
    masks_dir = scenario_cfg['masks_dir']

    model_image_size = MODEL_CONFIGS[model_type].get('image_size', IMAGE_SIZE)

    full_dataset = ForgeryDataset(
        root_dir=train_dir,
        masks_dir=masks_dir,
        transform=None,
        mode='train'
    )

    train_idx, val_idx, test_idx = stratified_train_val_test_split(
        full_dataset,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        seed=RANDOM_SEED
    )

    # Política assimétrica de transforms
    train_transform_heavy = get_transforms(model_image_size, mode='train')
    train_transform_light = get_transforms(model_image_size, mode='train_light')
    val_transform = get_transforms(model_image_size, mode='val')

    train_dataset = TransformSubset(full_dataset, train_idx, train_transform_heavy,
                                    transform_light=train_transform_light)
    val_dataset = TransformSubset(full_dataset, val_idx, val_transform)

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

    return train_loader, val_loader


def train_epoch_fast(model, dataloader, criterion, optimizer, device, scaler=None):
    """Treina uma época (versão simplificada sem progress bar para busca)."""
    model.train()
    losses = AverageMeter()
    correct = 0
    total = 0
    use_amp = scaler is not None

    for images, labels, _ in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast('cuda', enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        losses.update(loss.item(), images.size(0))

    return losses.avg, correct / total * 100


def validate_fast(model, dataloader, criterion, device, use_amp=False):
    """Valida o modelo (versão simplificada para busca)."""
    model.eval()
    losses = AverageMeter()
    all_labels = []
    all_predictions = []
    all_probs = []

    with torch.no_grad():
        for images, labels, _ in dataloader:
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

    metrics = calculate_metrics(
        np.array(all_labels),
        np.array(all_predictions),
        np.array(all_probs)
    )
    return losses.avg, metrics


# ============================================================================
# Treino rápido para avaliação de hiperparâmetros
# ============================================================================

def evaluate_config(model_type, config, train_loader, val_loader, device, trial_num, total_trials):
    """
    Treina um modelo com épocas reduzidas e retorna métricas de validação.
    Usado para avaliar uma configuração de hiperparâmetros.
    """
    search_epochs = SEARCH_EPOCHS[model_type]

    print(f"\n  Trial {trial_num}/{total_trials}: {config}")

    model = create_model(model_type, device)
    scaler = torch.amp.GradScaler('cuda') if USE_AMP else None

    # Criterion com class_weights do config
    class_weight_1 = config.get('class_weights_1', 1.0)
    weight = torch.tensor([1.0, class_weight_1], dtype=torch.float32).to(device)
    label_smoothing = config.get('label_smoothing', 0.05)
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)

    best_val_f1 = 0.0
    best_val_acc = 0.0
    best_metrics = {}

    # ===== Fase 1: Backbone congelado =====
    model.freeze_backbone()
    optimizer_p1 = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['phase1_lr'],
        weight_decay=config['weight_decay']
    )
    scheduler_p1 = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_p1, T_max=search_epochs['phase1']
    )

    for epoch in range(search_epochs['phase1']):
        train_epoch_fast(model, train_loader, criterion, optimizer_p1, device, scaler)
        scheduler_p1.step()

    # ===== Fase 2: Fine-tuning completo =====
    model.unfreeze_backbone()
    param_groups = model.get_param_groups(
        backbone_lr=config['phase2_backbone_lr'],
        classifier_lr=config['phase2_classifier_lr']
    )
    optimizer_p2 = optim.AdamW(param_groups, weight_decay=config['weight_decay'])

    warmup_epochs = 2
    phase2_epochs = search_epochs['phase2']

    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            return float(current_epoch + 1) / float(warmup_epochs)
        progress = float(current_epoch - warmup_epochs) / float(max(1, phase2_epochs - warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler_p2 = optim.lr_scheduler.LambdaLR(optimizer_p2, lr_lambda)

    for epoch in range(phase2_epochs):
        train_loss, train_acc = train_epoch_fast(
            model, train_loader, criterion, optimizer_p2, device, scaler
        )
        val_loss, val_metrics = validate_fast(
            model, val_loader, criterion, device, use_amp=USE_AMP
        )
        scheduler_p2.step()

        val_f1 = val_metrics['f1_score']
        val_acc = val_metrics['accuracy'] * 100

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_acc = val_acc
            best_metrics = val_metrics.copy()

    print(f"    → Val Acc: {best_val_acc:.2f}% | F1: {best_val_f1:.4f} | "
          f"AUC: {best_metrics.get('roc_auc', 0):.4f}")

    del model
    torch.cuda.empty_cache()

    return {
        'config': {k: float(v) if isinstance(v, (np.floating, float)) else v
                   for k, v in config.items()},
        'best_val_acc': float(best_val_acc),
        'best_val_f1': float(best_val_f1),
        'best_val_auc': float(best_metrics.get('roc_auc', 0)),
        'best_metrics': {k: float(v) if isinstance(v, (np.floating, float)) else v
                         for k, v in best_metrics.items()
                         if k != 'confusion_matrix'},
    }


# ============================================================================
# Search principal
# ============================================================================

def run_search(model_type, scenario, method='grid', n_iter=20):
    """
    Executa Grid Search ou Randomized Search para um modelo/cenário.

    Args:
        model_type: 'resnet50' ou 'dinov2'
        scenario: cenário de dados
        method: 'grid' ou 'random'
        n_iter: número de iterações para random search

    Returns:
        dict com resultados ordenados por F1-score
    """
    print(f"\n{'='*70}")
    print(f"HYPERPARAMETER SEARCH — {method.upper()}")
    print(f"Modelo: {MODEL_CONFIGS[model_type]['name']}  |  Cenário: {scenario}")
    print(f"{'='*70}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA não disponível.")
    device = torch.device('cuda')
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    set_seed(RANDOM_SEED)

    # Gerar configurações
    space = SEARCH_SPACES[model_type][method]
    if method == 'grid':
        configs = generate_grid_configs(space)
        print(f"\nGrid Search: {len(configs)} combinações")
    else:
        configs = generate_random_configs(space, n_iter)
        print(f"\nRandomized Search: {n_iter} combinações")

    # DataLoaders (criados uma vez, reutilizados para todas as configs)
    print(f"\nCarregando dataset ({scenario})...")
    train_loader, val_loader = create_dataloaders(scenario, model_type, BATCH_SIZE)

    search_epochs = SEARCH_EPOCHS[model_type]
    print(f"Épocas por trial: fase 1={search_epochs['phase1']}, fase 2={search_epochs['phase2']}")

    # Executar busca
    results = []
    search_start = time.time()

    for i, config in enumerate(configs, 1):
        trial_start = time.time()
        result = evaluate_config(
            model_type, config, train_loader, val_loader, device, i, len(configs)
        )
        result['trial_time'] = format_time(time.time() - trial_start)
        results.append(result)

    search_time = time.time() - search_start

    # Ordenar por F1-score (métrica principal)
    results.sort(key=lambda r: r['best_val_f1'], reverse=True)

    # Relatório
    print(f"\n{'='*70}")
    print(f"RESULTADOS — TOP 10")
    print(f"{'='*70}")
    print(f"{'#':<4} {'Acc':<8} {'F1':<8} {'AUC':<8} {'Tempo':<10} {'Config'}")
    print(f"{'─'*70}")

    for i, r in enumerate(results[:10], 1):
        cfg_str = ', '.join(f"{k}={v:.2e}" if isinstance(v, float) and v < 0.01
                           else f"{k}={v:.3f}" if isinstance(v, float)
                           else f"{k}={v}"
                           for k, v in r['config'].items())
        print(f"{i:<4} "
              f"{r['best_val_acc']:<8.2f} "
              f"{r['best_val_f1']:<8.4f} "
              f"{r['best_val_auc']:<8.4f} "
              f"{r['trial_time']:<10} "
              f"{cfg_str}")

    best = results[0]
    print(f"\n{'='*70}")
    print(f"MELHOR CONFIGURAÇÃO:")
    print(f"{'='*70}")
    for k, v in best['config'].items():
        if isinstance(v, float) and v < 0.01:
            print(f"  {k}: {v:.2e}")
        elif isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    print(f"\n  Val Acc: {best['best_val_acc']:.2f}%")
    print(f"  Val F1:  {best['best_val_f1']:.4f}")
    print(f"  Val AUC: {best['best_val_auc']:.4f}")
    print(f"\n  Tempo total de busca: {format_time(search_time)}")
    print(f"  Tempo médio por trial: {format_time(search_time / len(configs))}")

    # Salvar resultados
    os.makedirs(RESULTS_DIR, exist_ok=True)
    run_name = f"{model_type}_{scenario}"
    results_path = os.path.join(RESULTS_DIR, f'{run_name}_{method}_search.json')
    save_data = {
        'method': method,
        'model_type': model_type,
        'scenario': scenario,
        'n_configs': len(configs),
        'search_time': format_time(search_time),
        'search_epochs': search_epochs,
        'best_config': best['config'],
        'best_metrics': {
            'val_acc': best['best_val_acc'],
            'val_f1': best['best_val_f1'],
            'val_auc': best['best_val_auc'],
        },
        'all_results': results,
    }
    with open(results_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResultados salvos em: {results_path}")

    # Gerar config pronto para usar no FINETUNE_CONFIGS
    print(f"\n{'='*70}")
    print(f"CONFIG PARA configs/config.py (FINETUNE_CONFIGS['{model_type}']):")
    print(f"{'='*70}")
    bc = best['config']
    print(f"    'phase1_lr': {bc['phase1_lr']:.2e},")
    print(f"    'phase2_backbone_lr': {bc['phase2_backbone_lr']:.2e},")
    print(f"    'phase2_classifier_lr': {bc['phase2_classifier_lr']:.2e},")
    print(f"    'weight_decay': {bc['weight_decay']:.2e},")
    print(f"    'label_smoothing': {bc['label_smoothing']:.4f},")
    print(f"    # class_weights: [1.0, {bc['class_weights_1']:.2f}]")

    return save_data


def main():
    parser = argparse.ArgumentParser(
        description='Hyperparameter Search para modelos de detecção de anomalias'
    )
    parser.add_argument('--model', type=str, default='resnet50',
                       choices=['all', 'resnet50', 'dinov2'],
                       help='Modelo (default: resnet50)')
    parser.add_argument('--scenario', type=str, default='no_augmentation',
                       choices=['all', 'no_augmentation', 'no_synthetic', 'with_synthetic'],
                       help='Cenário (default: no_augmentation)')
    parser.add_argument('--method', type=str, default='grid',
                       choices=['grid', 'random'],
                       help='Método de busca (default: grid)')
    parser.add_argument('--n-iter', type=int, default=20,
                       help='Número de iterações para random search (default: 20)')

    args = parser.parse_args()

    models = ['resnet50', 'dinov2'] if args.model == 'all' else [args.model]
    scenarios = ['no_augmentation', 'no_synthetic', 'with_synthetic'] if args.scenario == 'all' else [args.scenario]

    all_results = {}
    total_start = time.time()

    for scenario in scenarios:
        for model_type in models:
            result = run_search(model_type, scenario, args.method, args.n_iter)
            key = f"{model_type}_{scenario}"
            all_results[key] = {
                'best_config': result['best_config'],
                'best_f1': result['best_metrics']['val_f1'],
                'best_acc': result['best_metrics']['val_acc'],
            }

    # Resumo final
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print(f"RESUMO DE TODAS AS BUSCAS")
        print(f"{'='*70}")
        print(f"{'Run':<32} {'Melhor Acc':<12} {'Melhor F1':<12}")
        print(f"{'─'*56}")
        for key, r in all_results.items():
            print(f"{key:<32} {r['best_acc']:<12.2f} {r['best_f1']:<12.4f}")
        print(f"\nTempo total: {format_time(time.time() - total_start)}")


if __name__ == '__main__':
    main()
