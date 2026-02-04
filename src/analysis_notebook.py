"""
Notebook interativo para análise e comparação dos resultados
Execute este arquivo no Jupyter ou como script Python
"""
# %% [markdown]
# # Laboratório de Detecção de Anomalias em Imagens
# ## Comparação entre CNN, ViT, CvT e DINOv2
# 
# Este notebook analisa e compara os resultados dos diferentes modelos treinados.

# %% Imports
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configurações de visualização
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

# %% [markdown]
# ## 1. Carregar Resultados

# %% Carregar dados
BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / 'results'

# Carregar históricos de treinamento
histories = {}
for model in ['resnet50', 'resnet101', 'vit', 'cvt13', 'cvt21', 'cvt_w24', 'dinov2']:
    history_file = RESULTS_DIR / f'{model}_history.json'
    if history_file.exists():
        with open(history_file, 'r') as f:
            histories[model] = json.load(f)

# Carregar resultados de teste
test_results = {}
for model in ['resnet50', 'resnet101', 'vit', 'cvt13', 'cvt21', 'cvt_w24', 'dinov2']:
    results_file = RESULTS_DIR / f'{model}_test_results.json'
    if results_file.exists():
        with open(results_file, 'r') as f:
            test_results[model] = json.load(f)

print(f"Históricos carregados: {list(histories.keys())}")
print(f"Resultados de teste carregados: {list(test_results.keys())}")

# %% [markdown]
# ## 2. Curvas de Treinamento

# %% Plot training curves
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

model_names = {
    'resnet50': 'ResNet-50',
    'resnet101': 'ResNet-101',
    'vit': 'Vision Transformer',
    'cvt13': 'CvT-13',
    'cvt21': 'CvT-21',
    'cvt_w24': 'CvT-W24',
    'dinov2': 'DINOv2'
}

# Loss
for model, history in histories.items():
    if 'train_loss' in history and 'val_loss' in history:
        axes[0].plot(history['train_loss'], label=f"{model_names[model]} (Train)", alpha=0.7)
        axes[0].plot(history['val_loss'], label=f"{model_names[model]} (Val)", linestyle='--')

axes[0].set_xlabel('Época')
axes[0].set_ylabel('Loss')
axes[0].set_title('Curvas de Loss Durante o Treinamento')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Accuracy
for model, history in histories.items():
    if 'train_acc' in history and 'val_acc' in history:
        axes[1].plot(history['train_acc'], label=f"{model_names[model]} (Train)", alpha=0.7)
        axes[1].plot(history['val_acc'], label=f"{model_names[model]} (Val)", linestyle='--')

axes[1].set_xlabel('Época')
axes[1].set_ylabel('Acurácia (%)')
axes[1].set_title('Curvas de Acurácia Durante o Treinamento')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS_DIR / 'training_curves_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 3. Comparação de Métricas de Teste

# %% Comparação de métricas
if test_results:
    metrics_names = ['accuracy', 'precision', 'recall', 'f1_score']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(metrics_names))
    width = 0.2
    
    for i, (model, results) in enumerate(test_results.items()):
        values = [results.get(metric, 0) * 100 for metric in metrics_names]
        ax.bar(x + i * width, values, width, label=model_names[model])
    
    ax.set_xlabel('Métricas')
    ax.set_ylabel('Score (%)')
    ax.set_title('Comparação de Desempenho dos Modelos no Conjunto de Teste')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(['Acurácia', 'Precisão', 'Recall', 'F1-Score'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 105])
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

# %% [markdown]
# ## 4. Tabela Resumo

# %% Criar tabela
if test_results:
    print("\n" + "="*80)
    print("TABELA RESUMO - DESEMPENHO DOS MODELOS")
    print("="*80)
    print(f"{'Modelo':<25} {'Acurácia':<12} {'Precisão':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-"*80)
    
    for model, results in test_results.items():
        print(f"{model_names[model]:<25} "
              f"{results['accuracy']*100:<12.2f} "
              f"{results['precision']*100:<12.2f} "
              f"{results['recall']*100:<12.2f} "
              f"{results['f1_score']*100:<12.2f}")
    
    print("="*80)

# %% [markdown]
# ## 5. Análise de Convergência

# %% Plot convergência
if histories:
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for model, history in histories.items():
        if 'val_acc' in history:
            val_accs = history['val_acc']
            # Calcular tendência (média móvel)
            window = 5
            if len(val_accs) >= window:
                moving_avg = np.convolve(val_accs, np.ones(window)/window, mode='valid')
                ax.plot(range(window-1, len(val_accs)), moving_avg, 
                       label=model_names[model], linewidth=2)
    
    ax.set_xlabel('Época')
    ax.set_ylabel('Acurácia de Validação (%)')
    ax.set_title('Convergência dos Modelos (Média Móvel de 5 épocas)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'convergence_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

# %% [markdown]
# ## 6. Análise Estatística

# %% Estatísticas
if test_results:
    print("\n" + "="*80)
    print("ANÁLISE DETALHADA")
    print("="*80)
    
    for model, results in test_results.items():
        print(f"\n{model_names[model]}:")
        print(f"  - Taxa de Verdadeiros Positivos: {results.get('true_positives', 0)}")
        print(f"  - Taxa de Verdadeiros Negativos: {results.get('true_negatives', 0)}")
        print(f"  - Taxa de Falsos Positivos: {results.get('false_positives', 0)}")
        print(f"  - Taxa de Falsos Negativos: {results.get('false_negatives', 0)}")
        
        if 'specificity' in results:
            print(f"  - Especificidade: {results['specificity']:.4f}")
        if 'roc_auc' in results:
            print(f"  - ROC-AUC: {results['roc_auc']:.4f}")
            print(f"  - PR-AUC: {results['pr_auc']:.4f}")

# %% [markdown]
# ## 7. Melhor Modelo

# %% Identificar melhor modelo
if test_results:
    best_model = max(test_results.items(), key=lambda x: x[1]['f1_score'])
    
    print("\n" + "="*80)
    print("RECOMENDAÇÃO")
    print("="*80)
    print(f"\n🏆 Melhor modelo baseado em F1-Score: {model_names[best_model[0]]}")
    print(f"\n   F1-Score: {best_model[1]['f1_score']:.4f}")
    print(f"   Acurácia: {best_model[1]['accuracy']*100:.2f}%")
    print(f"   Precisão: {best_model[1]['precision']:.4f}")
    print(f"   Recall: {best_model[1]['recall']:.4f}")
    print("\n" + "="*80)

# %% [markdown]
# ## 8. Gerar Relatório para o TCC

# %% Gerar relatório
report_path = RESULTS_DIR / 'relatorio_tcc.txt'

with open(report_path, 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("RELATÓRIO DE EXPERIMENTOS - DETECÇÃO DE ANOMALIAS EM IMAGENS\n")
    f.write("="*80 + "\n\n")
    
    f.write("1. MODELOS TESTADOS\n")
    f.write("-" * 80 + "\n")
    for model in model_names.values():
        f.write(f"   - {model}\n")
    
    f.write("\n2. RESULTADOS DE TESTE\n")
    f.write("-" * 80 + "\n")
    f.write(f"{'Modelo':<25} {'Acurácia':<12} {'Precisão':<12} {'Recall':<12} {'F1-Score':<12}\n")
    f.write("-" * 80 + "\n")
    
    for model, results in test_results.items():
        f.write(f"{model_names[model]:<25} "
                f"{results['accuracy']*100:<12.2f} "
                f"{results['precision']*100:<12.2f} "
                f"{results['recall']*100:<12.2f} "
                f"{results['f1_score']*100:<12.2f}\n")
    
    f.write("\n3. CONCLUSÕES\n")
    f.write("-" * 80 + "\n")
    if test_results:
        best = max(test_results.items(), key=lambda x: x[1]['f1_score'])
        f.write(f"Melhor modelo: {model_names[best[0]]}\n")
        f.write(f"F1-Score: {best[1]['f1_score']:.4f}\n\n")
        
        f.write("Observações:\n")
        f.write("- Transformers (ViT, CvT, DINOv2) geralmente superam CNN em tarefas complexas\n")
        f.write("- DINOv2 usa aprendizado auto-supervisionado, bom para poucos dados rotulados\n")
        f.write("- CvT combina convoluções com atenção, oferecendo bom trade-off\n")
    
    f.write("\n" + "="*80 + "\n")

print(f"\n✓ Relatório salvo em: {report_path}")

# %%
