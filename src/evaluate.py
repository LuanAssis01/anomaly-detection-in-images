"""
Script de avaliação dos modelos ResNet-50 e DINOv2 treinados
Suporta cenários com e sem dados sintéticos, com medição de tempo.
"""
import os
import sys
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import argparse

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from configs.config import *
from configs.config import USE_AMP, DECISION_THRESHOLD, SCENARIOS
from utils.dataset import ForgeryDataset, get_transforms, custom_collate_fn
from utils.metrics import calculate_metrics
from utils.visualization import plot_confusion_matrix, visualize_predictions
from models import CNNModel, DINOv2Model


def format_time(seconds):
    """Formata segundos em HH:MM:SS"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


def load_model(model_type: str, checkpoint_path: str, device):
    """Carrega modelo treinado"""
    if model_type == 'resnet50':
        model = CNNModel(
            num_classes=NUM_CLASSES,
            model_name=MODEL_CONFIGS[model_type]['model_name'],
            pretrained=False
        )
    elif model_type == 'dinov2':
        model = DINOv2Model(
            model_name=MODEL_CONFIGS['dinov2']['model_name'],
            num_classes=NUM_CLASSES
        )
    else:
        raise ValueError(f"Modelo desconhecido: {model_type}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Remover prefixo '_orig_mod.' adicionado por torch.compile()
    state_dict = checkpoint['model_state_dict']
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('_orig_mod.', '')
        cleaned_state_dict[new_key] = value

    model.load_state_dict(cleaned_state_dict)
    model = model.to(device)
    model.eval()

    print(f"Modelo carregado (acc treino: {checkpoint.get('val_acc', 0):.2f}%)")
    return model


def evaluate_model(model, dataloader, device, threshold=None):
    """Avalia modelo no dataset com threshold de decisão configurável e mede tempo"""
    if threshold is None:
        threshold = DECISION_THRESHOLD
    model.eval()

    all_labels = []
    all_predictions = []
    all_probs = []
    all_images = []

    use_amp = USE_AMP

    eval_start = time.time()

    with torch.no_grad():
        for images, labels, _ in tqdm(dataloader, desc='Avaliando'):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=use_amp):
                outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            predicted = (probs >= threshold).long()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            if len(all_images) < 16:
                all_images.extend(images.cpu())

    eval_time = time.time() - eval_start

    metrics = calculate_metrics(
        np.array(all_labels),
        np.array(all_predictions),
        np.array(all_probs)
    )
    metrics['threshold'] = threshold
    metrics['eval_time_seconds'] = round(eval_time, 2)
    metrics['eval_time_formatted'] = format_time(eval_time)

    return metrics, all_labels, all_predictions, all_images[:16]


def print_metrics(model_name: str, scenario: str, metrics: dict):
    """Imprime métricas de forma formatada"""
    print(f"\n{'='*60}")
    print(f"Resultados - {model_name}  |  Cenário: {scenario}")
    print(f"{'='*60}")
    print(f"Threshold:     {metrics.get('threshold', 0.5)}")
    print(f"Accuracy:      {metrics['accuracy']*100:.2f}%")
    print(f"Precision:     {metrics['precision']:.4f}")
    print(f"Recall:        {metrics['recall']:.4f}")
    print(f"F1-Score:      {metrics['f1_score']:.4f}")
    print(f"Specificity:   {metrics.get('specificity', 0):.4f}")

    if 'roc_auc' in metrics:
        print(f"ROC-AUC:       {metrics['roc_auc']:.4f}")
        print(f"PR-AUC:        {metrics['pr_auc']:.4f}")

    print(f"\nTempo de teste: {metrics['eval_time_formatted']}")

    print(f"\nMatriz de Confusão:")
    print(f"  TN: {metrics.get('true_negatives', 0):4d}  FP: {metrics.get('false_positives', 0):4d}")
    print(f"  FN: {metrics.get('false_negatives', 0):4d}  TP: {metrics.get('true_positives', 0):4d}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Avaliar modelos treinados')
    parser.add_argument('--model', type=str, default='all',
                      choices=['all', 'resnet50', 'dinov2'],
                      help='Modelo a avaliar (default: all)')
    parser.add_argument('--scenario', type=str, default='all',
                      choices=['all', 'no_augmentation', 'no_synthetic', 'with_synthetic'],
                      help='Cenário de dados (default: all = todos os 3 cenários)')
    parser.add_argument('--visualize', action='store_true',
                      help='Gerar visualizações (confusion matrix, predictions)')
    parser.add_argument('--threshold', type=float, default=None,
                      help=f'Threshold de decisão (default: {DECISION_THRESHOLD} do config)')

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA não disponível. Este projeto requer GPU para execução.")
    device = torch.device('cuda')
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    models_to_eval = ['resnet50', 'dinov2'] if args.model == 'all' else [args.model]
    scenarios = ['no_augmentation', 'no_synthetic', 'with_synthetic'] if args.scenario == 'all' else [args.scenario]

    all_results = {}

    for scenario in scenarios:
        scenario_cfg = SCENARIOS[scenario]
        train_dir = scenario_cfg['train_dir']

        for model_type in models_to_eval:
            run_name = f"{model_type}_{scenario}"
            checkpoint_path = os.path.join(CHECKPOINTS_DIR, f'{run_name}_best.pth')

            if not os.path.exists(checkpoint_path):
                print(f"\nCheckpoint não encontrado: {checkpoint_path}")
                continue

            print(f"\n>>> Avaliando {model_type} | cenário: {scenario}")

            # Dataset de teste (split val do cenário)
            model_image_size = MODEL_CONFIGS[model_type].get('image_size', IMAGE_SIZE)

            full_dataset = ForgeryDataset(
                root_dir=train_dir,
                masks_dir=scenario_cfg['masks_dir'],
                transform=get_transforms(model_image_size, mode='val'),
                mode='test'
            )

            from torch.utils.data import random_split
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
                pin_memory=True,
                persistent_workers=True if NUM_WORKERS > 0 else False,
                prefetch_factor=2 if NUM_WORKERS > 0 else None,
                collate_fn=custom_collate_fn
            )

            print(f"Samples de teste: {len(test_dataset)}")

            model = load_model(model_type, checkpoint_path, device)
            threshold = args.threshold if args.threshold is not None else DECISION_THRESHOLD
            metrics, labels, predictions, images = evaluate_model(model, test_loader, device, threshold=threshold)

            all_results[run_name] = metrics
            print_metrics(MODEL_CONFIGS[model_type]['name'], scenario, metrics)

            # Salvar métricas em JSON
            results_path = os.path.join(RESULTS_DIR, f'{run_name}_test_results.json')
            with open(results_path, 'w') as f:
                metrics_save = {k: v for k, v in metrics.items() if k != 'confusion_matrix'}
                json.dump(metrics_save, f, indent=2)

            # Visualizações
            if args.visualize:
                cm = np.array(metrics['confusion_matrix'])
                plot_confusion_matrix(
                    cm,
                    ['Authentic', 'Forged'],
                    save_path=os.path.join(RESULTS_DIR, f'{run_name}_confusion_matrix.png')
                )
                visualize_predictions(
                    torch.stack(images[:8]),
                    torch.tensor(labels[:8]),
                    torch.tensor(predictions[:8]),
                    num_images=8,
                    save_path=os.path.join(RESULTS_DIR, f'{run_name}_predictions.png')
                )

            del model
            torch.cuda.empty_cache()

    # Comparação geral
    if len(all_results) > 1:
        print(f"\n{'='*90}")
        print(f"COMPARAÇÃO GERAL")
        print(f"{'='*90}")
        print(f"{'Run':<30} {'Acc':<8} {'Prec':<8} {'Rec':<8} {'F1':<8} {'AUC':<8} {'Tempo':<12}")
        print(f"{'─'*90}")

        for run_name, metrics in all_results.items():
            print(f"{run_name:<30} "
                  f"{metrics['accuracy']*100:<8.2f} "
                  f"{metrics['precision']:<8.4f} "
                  f"{metrics['recall']:<8.4f} "
                  f"{metrics['f1_score']:<8.4f} "
                  f"{metrics.get('roc_auc', 0):<8.4f} "
                  f"{metrics['eval_time_formatted']:<12}")

        print(f"{'='*90}\n")

        # Salvar comparação
        comparison_path = os.path.join(RESULTS_DIR, 'evaluation_comparison.json')
        comparison = {}
        for run_name, metrics in all_results.items():
            comparison[run_name] = {
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score'],
                'roc_auc': metrics.get('roc_auc', 0),
                'eval_time_seconds': metrics['eval_time_seconds'],
                'eval_time_formatted': metrics['eval_time_formatted'],
            }
        with open(comparison_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        print(f"Comparação salva em: {comparison_path}")


if __name__ == '__main__':
    main()
