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
  ResNet-50: Imagem (384×384 ou 518×518) → Backbone → MLP (2048→BN→512→BN→256→2) → logits
  DINOv2:    Imagem (384×384 ou 518×518) → Backbone → MLP (768→LN→768→LN→384→2) → logits
  Loss: CrossEntropyLoss(weights=[1.0, 1.5], label_smoothing)
  Threshold de decisão: 0.4 (favorece recall)

  Nota: 518×518 é a resolução nativa do DINOv2 (patch_size=14, 518=14×37), evitando
  interpolação de positional embeddings. Ambas as resoluções foram experimentadas
  para comparação — ver seção RESULTADOS.

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
- BATCH_SIZE: 6 (efetivo 12 com GRADIENT_ACCUMULATION_STEPS=2) — reduzido para 518×518
  (era 8/efetivo 16 com 384×384)
- IMAGE_SIZE: 518 (resolução nativa DINOv2; 384 também foi experimentado)
- USE_AMP: True (Mixed Precision FP16)
- GRADIENT_ACCUMULATION_STEPS: 2
- DECISION_THRESHOLD: 0.4 (favorece recall, usado em evaluate.py e app.py)

ResNet-50 fine-tuning (otimizado via Grid/Randomized Search):
  Fase 1: 8 épocas, LR 6e-4 (backbone congelado)
  Fase 2: 40 épocas, backbone LR 7e-5, classifier LR 3e-4
  Warmup: 3 épocas, label smoothing: 0.05, weight_decay: 1e-3
  class_weights: [1.0, 1.5]

DINOv2 fine-tuning (otimizado via Grid/Randomized Search):
  Fase 1: 12 épocas, LR 5e-4 (backbone congelado)
  Fase 2: 35 épocas, backbone LR 2e-6, classifier LR 2e-4
  Warmup: 4 épocas, label smoothing: 0.05, weight_decay: 2e-3
  class_weights: [1.0, 1.5]
  ATENÇÃO: backbone_lr > 5e-6 causa colapso do DINOv2 (recall cai para ~0%)

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

Melhores resultados obtidos (IMAGE_SIZE=518, hiperparâmetros otimizados):
  resnet50_no_augmentation  — Acc: 78.83%  F1: 0.8094  AUC: 0.8841
  resnet50_no_synthetic     — Acc: 81.30%  F1: 0.8326  AUC: 0.9077
  resnet50_with_synthetic   — Acc: 78.31%  F1: 0.8069  AUC: 0.8867
  dinov2_no_augmentation    — Acc: 84.55%  F1: 0.8618  AUC: 0.9442
  dinov2_no_synthetic       — Acc: 85.32%  F1: 0.8650  AUC: 0.9499  ← melhor DINOv2
  dinov2_with_synthetic     — Acc: 84.94%  F1: 0.8632  AUC: 0.9382

Observações experimentais relevantes:
  - DINOv2 supera ResNet-50 em todas as métricas e cenários (~5pp F1, ~6pp AUC)
  - 518×518 melhora DINOv2 consistentemente vs 384×384 (+0.6-1.9pp F1)
  - 518×518 tem efeito misto no ResNet-50 (melhora no_synthetic, piora no_augmentation)
  - with_synthetic é o pior ou igual ao no_synthetic para ambos os modelos
    (forgeries sintéticas têm artefatos distintos das reais, prejudicando generalização)
  - Hiperparâmetros otimizados via Grid/Randomized Search (ver results/*_search.json)

⚠️ REQUISITOS COMPUTACIONAIS
-----------------------------
- GPU: 8GB+ VRAM (testado em RTX 3060 12GB)
- RAM: 16GB+
- Tempo estimado por run (RTX 3060, IMAGE_SIZE=518):
    ResNet-50: ~45-55 min por run
    DINOv2:    ~2h-2h30 por run
    Total 6 runs: ~9h

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
Última atualização: 2026-04-01
================================================================================