"""
Notebook interativo para análise e comparação dos resultados
ResNet-50 vs DINOv2 em cenários com e sem dados sintéticos
Execute este arquivo no Jupyter ou como script Python
"""
# %% [markdown]
# # Detecção de Anomalias em Imagens
# ## Comparação: ResNet-50 vs DINOv2 | Com vs Sem Dados Sintéticos

# %% Imports
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 10

# %% [markdown]
# ## 1. Carregar Resultados

# %% Carregar dados
BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / 'results'

MODELS = ['resnet50', 'dinov2']
SCENARIOS = ['no_augmentation', 'no_synthetic', 'with_synthetic']
MODEL_NAMES = {
    'resnet50': 'ResNet-50',
    'dinov2': 'DINOv2'
}
SCENARIO_NAMES = {
    'no_augmentation': 'Sem Augmentation',
    'no_synthetic': 'Com Augmentation',
    'with_synthetic': 'Com Sintéticas',
}

# Carregar históricos de treinamento
histories = {}
for model in MODELS:
    for scenario in SCENARIOS:
        run_name = f'{model}_{scenario}'
        history_file = RESULTS_DIR / f'{run_name}_history.json'
        if history_file.exists():
            with open(history_file, 'r') as f:
                histories[run_name] = json.load(f)

# Carregar resultados de teste
test_results = {}
for model in MODELS:
    for scenario in SCENARIOS:
        run_name = f'{model}_{scenario}'
        results_file = RESULTS_DIR / f'{run_name}_test_results.json'
        if results_file.exists():
            with open(results_file, 'r') as f:
                test_results[run_name] = json.load(f)

print(f"Históricos carregados: {list(histories.keys())}")
print(f"Resultados de teste:   {list(test_results.keys())}")

# %% [markdown]
# ## 2. Curvas de Treinamento

# %% Plot training curves
if histories:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    colors = {
        'resnet50_no_synthetic': '#1f77b4',
        'resnet50_with_synthetic': '#aec7e8',
        'dinov2_no_synthetic': '#ff7f0e',
        'dinov2_with_synthetic': '#ffbb78',
    }

    for run_name, history in histories.items():
        parts = run_name.split('_', 1)
        model = parts[0]
        scenario = parts[1]
        label = f"{MODEL_NAMES[model]} ({SCENARIO_NAMES[scenario]})"
        color = colors.get(run_name, None)

        if 'train_loss' in history:
            axes[0].plot(history['train_loss'], label=f"{label} (Train)", alpha=0.7, color=color)
            axes[0].plot(history['val_loss'], label=f"{label} (Val)", linestyle='--', color=color, alpha=0.5)

        if 'train_acc' in history:
            axes[1].plot(history['train_acc'], label=f"{label} (Train)", alpha=0.7, color=color)
            axes[1].plot(history['val_acc'], label=f"{label} (Val)", linestyle='--', color=color, alpha=0.5)

    axes[0].set_xlabel('Época')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Curvas de Loss')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Época')
    axes[1].set_ylabel('Acurácia (%)')
    axes[1].set_title('Curvas de Acurácia')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'training_curves_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

# %% [markdown]
# ## 3. Comparação de Métricas de Teste

# %% Comparação de métricas
if test_results:
    metrics_names = ['accuracy', 'precision', 'recall', 'f1_score']
    metric_labels = ['Acurácia', 'Precisão', 'Recall', 'F1-Score']

    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(metrics_names))
    n_runs = len(test_results)
    width = 0.8 / max(n_runs, 1)

    for i, (run_name, results) in enumerate(test_results.items()):
        parts = run_name.split('_', 1)
        model = parts[0]
        scenario = parts[1]
        label = f"{MODEL_NAMES[model]} ({SCENARIO_NAMES[scenario]})"
        values = [results.get(metric, 0) * 100 for metric in metrics_names]
        ax.bar(x + i * width, values, width, label=label)

    ax.set_xlabel('Métricas')
    ax.set_ylabel('Score (%)')
    ax.set_title('Comparação de Desempenho: Modelo x Cenário')
    ax.set_xticks(x + width * (n_runs - 1) / 2)
    ax.set_xticklabels(metric_labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 105])

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

# %% [markdown]
# ## 4. Tabela Resumo com Tempos

# %% Criar tabela
if test_results or histories:
    print("\n" + "=" * 100)
    print("TABELA RESUMO - DESEMPENHO E TEMPO")
    print("=" * 100)
    print(f"{'Run':<30} {'Acc':<10} {'Prec':<10} {'Rec':<10} {'F1':<10} {'Treino':<14} {'Teste':<12}")
    print("-" * 100)

    for model in MODELS:
        for scenario in SCENARIOS:
            run_name = f'{model}_{scenario}'
            label = f"{MODEL_NAMES[model]} ({SCENARIO_NAMES[scenario]})"

            results = test_results.get(run_name, {})
            history = histories.get(run_name, {})
            timing = history.get('timing', {})

            acc = f"{results.get('accuracy', 0)*100:.2f}" if results else "---"
            prec = f"{results.get('precision', 0)*100:.2f}" if results else "---"
            rec = f"{results.get('recall', 0)*100:.2f}" if results else "---"
            f1 = f"{results.get('f1_score', 0)*100:.2f}" if results else "---"
            train_t = timing.get('total_train_formatted', '---')
            eval_t = results.get('eval_time_formatted', '---')

            print(f"{label:<30} {acc:<10} {prec:<10} {rec:<10} {f1:<10} {train_t:<14} {eval_t:<12}")

    print("=" * 100)

# %% [markdown]
# ## 5. Análise Estatística

# %% Estatísticas
if test_results:
    print("\n" + "=" * 80)
    print("ANÁLISE DETALHADA")
    print("=" * 80)

    for run_name, results in test_results.items():
        parts = run_name.split('_', 1)
        model = parts[0]
        scenario = parts[1]
        label = f"{MODEL_NAMES[model]} ({SCENARIO_NAMES[scenario]})"

        print(f"\n{label}:")
        print(f"  - Verdadeiros Positivos:  {results.get('true_positives', 0)}")
        print(f"  - Verdadeiros Negativos:  {results.get('true_negatives', 0)}")
        print(f"  - Falsos Positivos:       {results.get('false_positives', 0)}")
        print(f"  - Falsos Negativos:       {results.get('false_negatives', 0)}")

        if 'specificity' in results:
            print(f"  - Especificidade:         {results['specificity']:.4f}")
        if 'roc_auc' in results:
            print(f"  - ROC-AUC:                {results['roc_auc']:.4f}")
            print(f"  - PR-AUC:                 {results['pr_auc']:.4f}")

# %% [markdown]
# ## 6. Melhor Modelo

# %% Identificar melhor modelo
if test_results:
    best_run = max(test_results.items(), key=lambda x: x[1]['f1_score'])
    parts = best_run[0].split('_', 1)
    best_label = f"{MODEL_NAMES[parts[0]]} ({SCENARIO_NAMES[parts[1]]})"

    print("\n" + "=" * 80)
    print("RECOMENDAÇÃO")
    print("=" * 80)
    print(f"\nMelhor modelo (F1-Score): {best_label}")
    print(f"  F1-Score:  {best_run[1]['f1_score']:.4f}")
    print(f"  Acurácia:  {best_run[1]['accuracy']*100:.2f}%")
    print(f"  Precisão:  {best_run[1]['precision']:.4f}")
    print(f"  Recall:    {best_run[1]['recall']:.4f}")
    print(f"  Tempo teste: {best_run[1].get('eval_time_formatted', '---')}")
    print("=" * 80)

# %% [markdown]
# ## 7. Relatório

# %% Gerar relatório
report_path = RESULTS_DIR / 'relatorio.txt'

with open(report_path, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("RELATÓRIO - DETECÇÃO DE ANOMALIAS EM IMAGENS\n")
    f.write("ResNet-50 vs DINOv2 | Com vs Sem Dados Sintéticos\n")
    f.write("=" * 80 + "\n\n")

    f.write("1. MODELOS TESTADOS\n")
    f.write("-" * 80 + "\n")
    for model in MODELS:
        f.write(f"   - {MODEL_NAMES[model]}\n")

    f.write("\n2. CENÁRIOS\n")
    f.write("-" * 80 + "\n")
    for scenario in SCENARIOS:
        f.write(f"   - {SCENARIO_NAMES[scenario]}\n")

    f.write("\n3. RESULTADOS\n")
    f.write("-" * 80 + "\n")
    f.write(f"{'Run':<30} {'Acc':<10} {'Prec':<10} {'Rec':<10} {'F1':<10} {'Tempo':<12}\n")
    f.write("-" * 80 + "\n")

    for model in MODELS:
        for scenario in SCENARIOS:
            run_name = f'{model}_{scenario}'
            results = test_results.get(run_name, {})
            label = f"{MODEL_NAMES[model]} ({SCENARIO_NAMES[scenario]})"
            if results:
                f.write(f"{label:<30} "
                        f"{results['accuracy']*100:<10.2f} "
                        f"{results['precision']*100:<10.2f} "
                        f"{results['recall']*100:<10.2f} "
                        f"{results['f1_score']*100:<10.2f} "
                        f"{results.get('eval_time_formatted', '---'):<12}\n")

    f.write("\n4. CONCLUSÕES\n")
    f.write("-" * 80 + "\n")
    if test_results:
        best = max(test_results.items(), key=lambda x: x[1]['f1_score'])
        parts = best[0].split('_', 1)
        f.write(f"Melhor modelo: {MODEL_NAMES[parts[0]]} ({SCENARIO_NAMES[parts[1]]})\n")
        f.write(f"F1-Score: {best[1]['f1_score']:.4f}\n\n")

        f.write("Observações:\n")
        f.write("- DINOv2 usa aprendizado auto-supervisionado, ideal para features visuais complexas\n")
        f.write("- ResNet-50 é mais rápido em treino e inferência, mas com menor capacidade discriminativa\n")
        f.write("- A comparação com/sem sintéticas mostra o impacto de data augmentation avançada\n")

    f.write("\n" + "=" * 80 + "\n")

print(f"\nRelatório salvo em: {report_path}")

# %%
