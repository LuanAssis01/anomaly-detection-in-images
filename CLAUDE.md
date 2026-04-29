================================================================================
LABORATÓRIO DE DETECÇÃO DE ANOMALIAS EM IMAGENS - TCC
================================================================================

🎯 OBJETIVO
-----------
Comparar CNNs (ResNet) vs Vision Transformers auto-supervisionados (DINOv2, DINOv3)
para detecção de falsificações em imagens, em 3 cenários de dados, gerando resultados
experimentais para artigo científico de TCC.

📊 MODELOS IMPLEMENTADOS
-------------------------
1. ResNet-50       — CNN clássica pré-treinada no ImageNet (baseline)
2. ResNet-101      — CNN maior, mesma abordagem
3. DINOv2-Base     — Self-supervised ViT (facebook/dinov2-base)
4. DINOv2-Large    — ViT maior, 307M params
5. DINOv3-Small    — ViT-S/16 treinado no LVD-1689M (facebook/dinov3-vits16-pretrain-lvd1689m)
6. DINOv3-Base     — ViT-B/16 treinado no LVD-1689M (facebook/dinov3-vitb16-pretrain-lvd1689m)
7. DINOv3-Large    — ViT-L/16 treinado no LVD-1689M (facebook/dinov3-vitl16-pretrain-lvd1689m)

📋 CENÁRIOS EXPERIMENTAIS
--------------------------
1. no_augmentation — apenas imagens originais (sem augmentation, sem sintéticas)
2. no_synthetic    — imagens originais + augmentation offline para balancear classes
3. with_synthetic  — originais + augmentation + forjadas sintéticas geradas de autênticas

Total atual: 7 modelos × 3 cenários = 21 runs experimentais possíveis.

🗂️ ESTRUTURA DO PROJETO
------------------------
anomaly-detection-in-images/
├── data/                        # Dataset de imagens
│   ├── train_images/            # Fonte original
│   │   ├── authentic/           # Imagens autênticas
│   │   └── forged/              # Imagens falsificadas
│   ├── train_masks/             # Máscaras de segmentação (.npy)
│   ├── test_images/             # Imagens de teste avulso
│   ├── supplemental_images/     # Imagens suplementares do dataset
│   ├── supplemental_masks/      # Máscaras suplementares (.npy)
│   ├── sample_submission.csv    # Exemplo de submissão
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
│   ├── compare_checkpoint_configs.py  # Compara configs entre checkpoints
│   └── app.py                   # Demo interativa
├── models/                      # Implementações dos modelos
│   ├── cnn_model.py             # ResNet-50/101 (backbone + classifier MLP)
│   ├── dino_model.py            # DINOv2Model + DINOv3Model (HuggingFace + classifier)
│   └── __init__.py
├── utils/                       # Utilitários
│   ├── dataset.py               # Dataset, transforms, split estratificado
│   ├── metrics.py               # Métricas de avaliação
│   ├── visualization.py         # Visualizações
│   └── __init__.py
├── configs/
│   └── config.py                # Parâmetros globais
├── results/                     # Resultados dos experimentos
├── checkpoints/                 # Modelos treinados + splits (IMAGE_SIZE=518)
│   ├── resnet50_<cenario>_best.pth
│   ├── resnet101_<cenario>_best.pth
│   ├── dinov2_<cenario>_best.pth
│   ├── dinov2_large_<cenario>_best.pth
│   ├── dinov3_<cenario>_best.pth        # DINOv3-Base
│   ├── dinov3_small_<cenario>_best.pth
│   ├── dinov3_large_<cenario>_best.pth
│   └── *_split.json                     # Índices do split por modelo/cenário
├── checkpoints_global_params_image_size_384/  # Backup: checkpoints com IMAGE_SIZE=384
├── checkpoints_with_ajusted_params/           # Backup: checkpoints com params ajustados
├── checkpoints_image_size_384.tar.xz          # Arquivo compactado dos checkpoints 384
├── venv/                        # Ambiente virtual Python
├── test.py                      # Teste de ambiente
├── requirements.txt             # Dependências
├── README.md                    # Documentação pública
├── start.sh / start.bat         # Scripts de inicialização
└── CLAUDE.md                    # Este arquivo

🏗️ ARQUITETURA DE TREINAMENTO
-------------------------------
Fine-tuning em 2 fases:
  Fase 1: Backbone congelado — treina apenas o classificador MLP
  Fase 2: Fine-tuning completo — backbone + classificador com LR diferenciado,
           warmup linear + cosine annealing, early stopping

Pipeline de dados:
  ResNet-50/101: Imagem → Backbone → MLP (2048→BN→512→BN→256→2) → logits
  DINOv2:        Imagem → Backbone → MLP (768→LN→768→LN→384→2) → logits
  DINOv3-Small:  Imagem → Backbone → MLP (384→LN→384→LN→192→2) → logits
  DINOv3-Base:   Imagem → Backbone → MLP (768→LN→768→LN→384→2) → logits
  DINOv3-Large:  Imagem → Backbone → MLP (1024→LN→1024→LN→512→2) → logits
  Loss: CrossEntropyLoss(weights=[1.0, 1.5], label_smoothing)
  Threshold de decisão: 0.4 (favorece recall)

  Nota sobre resolução:
  - DINOv2 patch_size=14: resolução nativa 518×518 (14×37), evita interpolação
  - DINOv3 patch_size=16: qualquer resolução funciona com interpolate_pos_encoding=True
  - DINOv3 carregado via AutoModel (suporta small/base/large automaticamente)

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
   python src/train.py --model dinov2 --scenario with_synthetic
   python src/train.py --model dinov3 --scenario no_augmentation
   python src/train.py --model dinov3_all --scenario all   # Só DINOv3, todos os cenários
   python src/train.py --model all --scenario all           # Todos os modelos e cenários

5. HYPERPARAMETER SEARCH (opcional, antes de treinar):
   python src/hyperparam_search.py --model resnet50 --scenario no_augmentation --method grid
   python src/hyperparam_search.py --model dinov2 --scenario no_augmentation --method random --n-iter 20
   python src/hyperparam_search.py --model dinov3_all --scenario all --method random --n-iter 20
   python src/hyperparam_search.py --model all --scenario all --method grid

6. AVALIAR MODELOS:
   python src/evaluate.py --model dinov3_all --scenario all --visualize
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
- BATCH_SIZE: 6 (efetivo 12 com GRADIENT_ACCUMULATION_STEPS=2)
- IMAGE_SIZE: 518 (resolução nativa DINOv2, patch_size=14, evita interpolação; melhores resultados com DINOv2-Large)
- USE_AMP: True (Mixed Precision FP16)
- GRADIENT_ACCUMULATION_STEPS: 2
- DECISION_THRESHOLD: 0.4 (favorece recall, usado em evaluate.py e app.py)

ResNet-50 fine-tuning (otimizado via Grid/Randomized Search):
  Fase 1: 8 épocas, LR 6e-4 (backbone congelado)
  Fase 2: 40 épocas, backbone LR 7e-5, classifier LR 3e-4
  Warmup: 3 épocas, label smoothing: 0.05, weight_decay: 1e-3
  class_weights: [1.0, 1.5]

ResNet-101 fine-tuning (valores conservadores — modelo maior que ResNet-50):
  Fase 1: 8 épocas, LR 6e-4 (backbone congelado)
  Fase 2: 40 épocas, backbone LR 5e-5, classifier LR 3e-4
  Warmup: 3 épocas, label smoothing: 0.05, weight_decay: 1e-3
  class_weights: [1.0, 1.5]

DINOv2-Base fine-tuning (otimizado via Grid/Randomized Search):
  Fase 1: 12 épocas, LR 5e-4 (backbone congelado)
  Fase 2: 35 épocas, backbone LR 2e-6, classifier LR 2e-4
  Warmup: 4 épocas, label smoothing: 0.05, weight_decay: 2e-3
  class_weights: [1.0, 1.5]
  batch_size: 4 (reduzido para caber na VRAM com 518×518)
  ATENÇÃO: backbone_lr > 5e-6 causa colapso do DINOv2 (recall cai para ~0%)

DINOv2-Large fine-tuning (mais conservador que Base — 307M params):
  Fase 1: 12 épocas, LR 3e-4 (backbone congelado)
  Fase 2: 35 épocas, backbone LR 1e-6, classifier LR 2e-4
  Warmup: 5 épocas, label smoothing: 0.05, weight_decay: 2e-3
  class_weights: [1.0, 1.5]
  batch_size: 4 (reduzido para caber na VRAM com 518×518)

DINOv3 fine-tuning (valores iniciais conservadores — otimizar via hyperparam_search):
  DINOv3-Small: Fase 1: 10 épocas LR 6e-4 | Fase 2: 35 épocas backbone 3e-6, classifier 2e-4
  DINOv3-Base:  Fase 1: 12 épocas LR 5e-4 | Fase 2: 35 épocas backbone 2e-6, classifier 2e-4
  DINOv3-Large: Fase 1: 12 épocas LR 3e-4 | Fase 2: 35 épocas backbone 1e-6, classifier 2e-4
                batch_size: 4 (ViT-L — reduzido para VRAM)
  ATENÇÃO: mesmo risco de colapso do DINOv2 — manter backbone_lr abaixo de 5e-6

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

Observações experimentais relevantes (ResNet/DINOv2):
  - DINOv2 supera ResNet-50 em todas as métricas e cenários (~5pp F1, ~6pp AUC)
  - 518×518 melhora DINOv2 consistentemente vs 384×384 (+0.6-1.9pp F1)
  - 518×518 tem efeito misto no ResNet-50 (melhora no_synthetic, piora no_augmentation)
  - with_synthetic é o pior ou igual ao no_synthetic para ambos os modelos
    (forgeries sintéticas têm artefatos distintos das reais, prejudicando generalização)
  - Hiperparâmetros otimizados via Grid/Randomized Search (ver results/*_search.json)

DINOv3 — resultados pendentes (modelos adicionados em 2026-04-10):

  - Rodar hyperparam_search antes do treino completo (backbone_lr é crítico)
  - Comparar small/base/large para custo-benefício de VRAM vs performance

🏆 RESULTADOS E CONCLUSÃO (FOCO DO ARTIGO)
-------------------------------------------

### Modelos principais (análise detalhada no artigo)

**ResNet-50** — baseline CNN:
  no_augmentation  — Acc: 78.83%  F1: 0.8094  AUC: 0.8841
  no_synthetic     — Acc: 81.30%  F1: 0.8326  AUC: 0.9077  ← melhor cenário
  with_synthetic   — Acc: 78.31%  F1: 0.8069  AUC: 0.8867

**DINOv2-Base** — Vision Transformer auto-supervisionado:
  no_augmentation  — Acc: 84.55%  F1: 0.8618  AUC: 0.9442
  no_synthetic     — Acc: 85.32%  F1: 0.8650  AUC: 0.9499  ← melhor cenário
  with_synthetic   — Acc: 84.94%  F1: 0.8632  AUC: 0.9382

**DINOv2-Large** — ViT-L/14, 307M params:
  no_augmentation  — pendente
  no_synthetic     — pendente
  with_synthetic   — pendente

### Modelos secundários (menção breve no artigo)
  ResNet-101    — variante maior do baseline; resultados pendentes
  DINOv3-Small  — resultados pendentes
  DINOv3-Base   — resultados pendentes
  DINOv3-Large  — resultados pendentes

### Conclusões parciais (ResNet-50 vs DINOv2-Base)

1. DINOv2-Base supera ResNet-50 em todos os cenários:
   - F1: +5.2 a +5.6 pontos percentuais
   - AUC: +5.7 a +6.0 pontos percentuais
   Demonstra a superioridade de representações auto-supervisionadas para detecção
   de falsificações em relação a CNNs supervisionadas no ImageNet.

2. Cenário no_synthetic é o melhor para ambos os modelos.
   O augmentation offline melhora a generalização sem introduzir artefatos espúrios.

3. with_synthetic piora ou iguala no_synthetic em ambos os modelos.
   Forgeries sintéticas apresentam padrões distintos das falsificações reais,
   prejudicando a generalização no conjunto de teste (composto apenas por imagens originais).

4. Impacto da resolução 518×518:
   - DINOv2: melhora consistente vs 384×384 (+0.6–1.9pp F1) — patch_size=14 nativo
   - ResNet-50: efeito misto (melhora no_synthetic, neutro/piora nos demais)

5. DINOv2-Large (resultado esperado após treinamento):
   Espera-se ganho adicional sobre DINOv2-Base dado o maior número de parâmetros
   (307M vs ~86M), porém com custo computacional ~35% maior e batch_size reduzido.

⚠️ REQUISITOS COMPUTACIONAIS
-----------------------------
- GPU: 8GB+ VRAM (testado em RTX 3060 12GB)
- RAM: 16GB+
- Tempo estimado por run (RTX 3060, IMAGE_SIZE=518):
    ResNet-50/101: ~45-55 min por run
    DINOv2-Base:   ~2h-2h30 por run
    DINOv2-Large:  ~3h+ por run (307M params, batch_size=4)
    DINOv3-Small:  ~1h30-2h por run
    DINOv3-Base:   ~2h-2h30 por run
    DINOv3-Large:  ~3h+ por run (batch_size=4)
    Total 21 runs estimado: ~40h+

📄 METODOLOGIA (ARTIGO)
-----------------------

### 1. Dataset
Utiliza-se o dataset CASIA (ou equivalente) composto por imagens digitais divididas em
duas classes: autênticas e falsificadas. As falsificações abrangem técnicas como
copy-move, splicing e inpainting. O dataset é particionado em três conjuntos de forma
estratificada (70% treino / 15% validação / 15% teste), garantindo que imagens
aumentadas/sintéticas pertençam exclusivamente ao treino (sem data leakage).

### 2. Cenários experimentais
Três cenários são avaliados para analisar o impacto da quantidade e qualidade dos dados:

- **no_augmentation**: apenas imagens originais, sem nenhuma transformação extra.
- **no_synthetic**: imagens originais + augmentation offline para balancear classes
  (horizontal flip para autênticas; rotação, brightness, zoom, shear para forjadas).
- **with_synthetic**: originais + augmentation + imagens forjadas sintéticas geradas
  a partir de autênticas por técnicas de copy-move, splicing, inpainting, noise injection
  e brightness manipulation.

### 3. Modelos
Sete modelos são comparados em dois grupos:

**CNNs (baseline):**
- ResNet-50 — backbone ResNet pré-treinado no ImageNet, classificador MLP (2048→512→256→2)
- ResNet-101 — variante maior da mesma família

**Vision Transformers auto-supervisionados:**
- DINOv2-Base  — ViT-B/14 treinado por destilação self-supervised (facebook/dinov2-base)
- DINOv2-Large — ViT-L/14, 307M parâmetros (facebook/dinov2-large)
- DINOv3-Small — ViT-S/16 treinado no LVD-1689M (facebook/dinov3-vits16-pretrain-lvd1689m)
- DINOv3-Base  — ViT-B/16 treinado no LVD-1689M (facebook/dinov3-vitb16-pretrain-lvd1689m)
- DINOv3-Large — ViT-L/16 treinado no LVD-1689M (facebook/dinov3-vitl16-pretrain-lvd1689m)

Todos os modelos recebem uma cabeça de classificação MLP treinada sobre o token [CLS].

### 4. Pré-processamento e augmentation
Todas as imagens são redimensionadas para **518×518 pixels** (resolução nativa do DINOv2,
patch_size=14, evitando interpolação de positional embeddings). Durante o treino, aplica-se
augmentation simétrica por classe (flips H/V, rotação ±10°, ColorJitter leve) para evitar
correlação espúria onde o modelo aprende "distorção = forjado". Validação e teste recebem
apenas resize + normalização ImageNet.

### 5. Fine-tuning em duas fases
O treinamento segue um protocolo de fine-tuning em duas fases para todos os modelos:

- **Fase 1** — backbone congelado: treina apenas o classificador MLP com LR mais alto,
  permitindo que a cabeça convirja antes de perturbar os pesos pré-treinados.
- **Fase 2** — fine-tuning completo: backbone + classificador com learning rates
  diferenciados (backbone_lr << classifier_lr), warmup linear + cosine annealing,
  early stopping (paciência=12 épocas).

Loss: CrossEntropyLoss com pesos de classe [1.0, 1.5] e label smoothing (0.05).
Threshold de decisão: 0.4 (favorece recall — detectar falsificações é mais crítico
do que reduzir falsos positivos).

### 6. Otimização de hiperparâmetros
Hiperparâmetros críticos (phase1_lr, phase2_backbone_lr, phase2_classifier_lr,
weight_decay, label_smoothing, class_weights) são otimizados via Grid Search e
Randomized Search usando épocas reduzidas para avaliação rápida, com resultados
ordenados por F1-score.

**ATENÇÃO**: backbone_lr > 5e-6 causa colapso dos modelos DINOv2/DINOv3 (recall ~0%).

### 7. Métricas de avaliação
- Acurácia, Precisão, Recall, F1-Score, Especificidade
- ROC-AUC, PR-AUC
- Matriz de Confusão
- Análise de threshold (sweep 0.30–0.70)

### 8. Infraestrutura
- GPU: NVIDIA RTX 3060 12GB
- Mixed Precision FP16 (USE_AMP=True)
- Gradient Accumulation (steps=2, batch efetivo=12)
- Framework: PyTorch + HuggingFace Transformers

📚 REFERÊNCIAS
--------------
- ResNet: "Deep Residual Learning" (He et al., 2015)
- DINOv2: "DINOv2: Learning Robust Visual Features" (Oquab et al., 2023)
- DINOv3: treinado no LVD-1689M — modelo não publicado formalmente, disponível via HuggingFace (facebook/dinov3-*)

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
Última atualização: 2026-04-23
================================================================================
