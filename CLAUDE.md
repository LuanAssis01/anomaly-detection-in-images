================================================================================
LABORATÓRIO DE DETECÇÃO DE ANOMALIAS EM IMAGENS - TCC
================================================================================

🎯 OBJETIVO
-----------
Comparar ResNet-50 (CNN) vs DINOv2 (Vision Transformer auto-supervisionado) para
detecção de falsificações em imagens, em 3 cenários de dados, gerando resultados
experimentais para artigo científico de TCC.

📊 MODELOS IMPLEMENTADOS
-------------------------
1. ResNet-50  — CNN clássica pré-treinada no ImageNet (baseline)
2. DINOv2     — Self-supervised Vision Transformer (facebook/dinov2-base)

📋 CENÁRIOS EXPERIMENTAIS
--------------------------
1. no_augmentation — apenas imagens originais (sem augmentation, sem sintéticas)
2. no_synthetic    — imagens originais + augmentation offline para balancear classes
3. with_synthetic  — originais + augmentation + forjadas sintéticas geradas de autênticas

Total: 2 modelos × 3 cenários = 6 runs experimentais.

🗂️ ESTRUTURA DO PROJETO
------------------------
anomaly-detection-in-images/
├── data/                        # Dataset de imagens
│   ├── train_images/            # Fonte original
│   │   ├── authentic/           # Imagens autênticas  
│   │   └── forged/              # Imagens falsificadas
│   ├── train_masks/             # Máscaras de segmentação (.npy)
│   └── scenarios/               # Datasets por cenário (gerados)
│       ├── no_augmentation/
│       ├── no_synthetic/
│       └── with_synthetic/
├── src/                         # Scripts principais
│   ├── train.py                 # Treinamento (2 fases fine-tuning)
│   ├── evaluate.py              # Avaliação e métricas
│   ├── generate_data.py         # Geração de dados por cenário
│   ├── hyperparam_search.py     # Grid Search + Randomized Search
│   ├── analysis_notebook.py     # Análise comparativa
│   └── app.py                   # Demo interativa
├── models/                      # Implementações dos modelos
│   ├── cnn_model.py             # ResNet-50 (backbone + classifier MLP)
│   └── dino_model.py            # DINOv2 (HuggingFace + classifier)
├── utils/                       # Utilitários
│   ├── dataset.py               # Dataset, transforms, split estratificado
│   ├── metrics.py               # Métricas de avaliação
│   └── visualization.py         # Visualizações
├── configs/
│   └── config.py                # Parâmetros globais
├── results/                     # Resultados dos experimentos
├── checkpoints/                 # Modelos treinados + splits
├── test.py                      # Teste de ambiente
├── requirements.txt             # Dependências
├── start.sh / start.bat         # Scripts de inicialização
└── CLAUDE.md                    # Este arquivo

🏗️ ARQUITETURA DE TREINAMENTO
-------------------------------
Fine-tuning em 2 fases:
  Fase 1: Backbone congelado — treina apenas o classificador MLP
  Fase 2: Fine-tuning completo — backbone + classificador com LR diferenciado,
           warmup linear + cosine annealing, early stopping

Pipeline de dados:
  ResNet-50: Imagem (384×384) → Backbone → MLP (2048→BN→512→BN→256→2) → logits
  DINOv2:    Imagem (384×384) → Backbone → MLP (768→LN→768→LN→384→2) → logits
  Loss: CrossEntropyLoss(weights=[1.0, 1.5], label_smoothing)
  Threshold de decisão: 0.4 (favorece recall)

Augmentation no runtime (mesma política para todas as imagens de treino):
  - flips horizontal/vertical + rotação ±10° + ColorJitter leve
  - Aplicada igualmente a autênticas e forjadas — evita correlação espúria
    onde o modelo aprende "distorção = forjado" em vez de artefatos reais
  - Val e test sem augmentation (só resize + normalize)

Geração offline (generate_data.py):
  - Autênticas: augmentation mínima (apenas horizontal flip)
  - Forjadas: augmentation agressiva (rotação ±20°, brightness, zoom, shear)

Split 3-way estratificado:
  - Originais divididas em train/val/test (70%/15%/15%)
  - Augmentadas/sintéticas sempre vão para treino
  - Val e test contêm apenas imagens originais (sem data leakage)

🚀 INÍCIO RÁPIDO
----------------

1. INSTALAR DEPENDÊNCIAS:
   pip install -r requirements.txt

2. TESTAR AMBIENTE:
   python test.py

3. GERAR DADOS POR CENÁRIO:
   python src/generate_data.py --scenario no_augmentation
   python src/generate_data.py --scenario no_synthetic --target 3000
   python src/generate_data.py --scenario with_synthetic --target 3000

4. TREINAR MODELOS:
   python src/train.py --model resnet50 --scenario no_augmentation
   python src/train.py --model resnet50 --scenario no_synthetic
   python src/train.py --model dinov2 --scenario with_synthetic
   python src/train.py --model all --scenario all    # Todos os 6 runs

5. HYPERPARAMETER SEARCH (opcional, antes de treinar):
   python src/hyperparam_search.py --model resnet50 --scenario no_augmentation --method grid
   python src/hyperparam_search.py --model dinov2 --scenario no_augmentation --method random --n-iter 20
   python src/hyperparam_search.py --model all --scenario all --method grid

6. AVALIAR MODELOS:
   python src/evaluate.py --model all --scenario all --visualize

7. DEMO INTERATIVA:
   python src/app.py

📈 MÉTRICAS AVALIADAS
---------------------
- Acurácia, Precisão, Recall, F1-Score, Especificidade
- ROC-AUC, PR-AUC
- Matriz de Confusão
- Análise de threshold (sweep 0.30–0.70)

🔍 HYPERPARAMETER SEARCH (src/hyperparam_search.py)
----------------------------------------------------
Otimização sistemática de hiperparâmetros com 2 métodos:

Grid Search: busca exaustiva sobre grade definida de hiperparâmetros.
  Parâmetros buscados: phase1_lr, phase2_backbone_lr, phase2_classifier_lr,
  weight_decay, label_smoothing, class_weights.

Randomized Search: amostragem aleatória de distribuições (log-uniform, uniform).
  Mais eficiente para espaços grandes. Default: 20 iterações.

Funcionamento:
  - Usa épocas reduzidas (fase1=3-4, fase2=8) para avaliação rápida
  - DataLoaders criados uma vez e reutilizados
  - Ordena resultados por F1-score
  - Gera config pronto para colar no FINETUNE_CONFIGS
  - Salva resultados em results/<modelo>_<cenario>_<metodo>_search.json

🛠️ CONFIGURAÇÕES PRINCIPAIS (configs/config.py)
-------------------------------------------------
- BATCH_SIZE: 8 (efetivo 16 com GRADIENT_ACCUMULATION_STEPS=2)
- IMAGE_SIZE: 384 (resolução alta para forgeries sutis)
- USE_AMP: True (Mixed Precision FP16)
- GRADIENT_ACCUMULATION_STEPS: 2 (batch efetivo = 8 × 2 = 16)
- DECISION_THRESHOLD: 0.4 (favorece recall, usado em evaluate.py e app.py)

ResNet-50 fine-tuning:
  Fase 1: 8 épocas, LR 5e-4 (backbone congelado)
  Fase 2: 40 épocas, backbone LR 2e-5, classifier LR 2e-4
  Warmup: 3 épocas, label smoothing: 0.05, weight_decay: 1e-4
  class_weights: [1.0, 1.0] — neutro, previne colapso para forged

DINOv2 fine-tuning:
  Fase 1: 12 épocas, LR 5e-4 (backbone congelado)
  Fase 2: 35 épocas, backbone LR 5e-6, classifier LR 5e-5
  Warmup: 4 épocas, label smoothing: 0.05, weight_decay: 0.05
  class_weights: [1.0, 1.5] — penaliza mais FN de forged

Data augmentation (runtime, mesma política para ambas as classes):
  flips (H+V), rotação ±10°, ColorJitter leve
  Sem GaussianBlur/GaussianNoise/RandomErasing — criam distribuição
  treino/teste muito diferente e causam colapso do modelo.

  Geração offline (generate_data.py):
    Autênticas: apenas horizontal flip (AUTHENTIC_AUGMENTATION)
    Forjadas: augmentation agressiva com máscaras sincronizadas (FORGED_AUGMENTATION)

📝 RESULTADOS
--------------
Salvos automaticamente em:
- results/<modelo>_<cenario>_history.json    # Histórico de treinamento
- results/<modelo>_<cenario>_test_results.json
- results/*_confusion_matrix.png
- checkpoints/<modelo>_<cenario>_best.pth    # Melhor modelo
- checkpoints/<modelo>_<cenario>_split.json  # Índices do split

⚠️ REQUISITOS COMPUTACIONAIS
-----------------------------
- GPU: 8GB+ VRAM (RTX 3070 ou superior)
- RAM: 16GB+
- Tempo estimado: ~10-15 min por run (6 runs no total)

📚 REFERÊNCIAS
--------------
- ResNet: "Deep Residual Learning" (He et al., 2015)
- DINOv2: "DINOv2: Learning Robust Visual Features" (Oquab et al., 2023)

🆘 TROUBLESHOOTING
------------------
- CUDA out of memory: Reduza BATCH_SIZE em config.py
- Modelo colapsando (recall ~0%): Augmentation assimétrica por classe cria
  correlação espúria (modelo aprende "distorção = forjado"). Manter a mesma
  augmentation para ambas as classes durante o treino.
- Cenários idênticos: Se o dataset original já tem >= target imagens por classe,
  no_synthetic e with_synthetic ficam iguais. Use --target maior ao gerar dados.
- Dataset vazio: Rodar generate_data.py antes de treinar

================================================================================
Última atualização: 2026-03-29
================================================================================