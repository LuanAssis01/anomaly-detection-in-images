# Detecção de Imagens Forjadas com Deep Learning

Projeto de detecção de manipulação em imagens utilizando modelos de deep learning (CNN, ViT, CvT e DINOv2).

## 📋 Descrição

Este projeto implementa e compara diferentes arquiteturas de redes neurais para detectar imagens forjadas (manipuladas) versus autênticas. O sistema classifica imagens em duas categorias:
- **Authentic**: Imagens originais sem manipulação
- **Forged**: Imagens com manipulação/edição digital

## 🏗️ Modelos Implementados

- **CNN (ResNet-50)**: Rede convolucional clássica
- **ViT (Vision Transformer)**: Transformer para visão computacional
- **CvT (Convolutional Vision Transformer)**: Híbrido CNN + Transformer
- **DINOv2**: Self-supervised vision transformer

## 🚀 Instalação

1. Clone o repositório:
```bash
git clone <seu-repositorio>
cd cvt-test
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

## 📊 Estrutura dos Dados

Organize seus dados na seguinte estrutura:
```
data/
├── train_images/
│   ├── authentic/
│   └── forged/
├── test_images/
├── train_masks/
└── sample_submission.csv
```

## 🎯 Uso

### Treinamento

Para treinar um modelo:
```bash
python src/train.py --model resnet50 --epochs 20 --batch-size 16
```

Modelos disponíveis: `resnet50`, `vit`, `cvt`, `dinov2`

### Avaliação

Para avaliar um modelo treinado:
```bash
python src/evaluate.py --model resnet50 --checkpoint checkpoints/resnet50_best.pth
```

### Interface Web

Execute a aplicação web para testar o modelo:
```bash
python src/app.py
```

## ⚙️ Configurações

Edite `configs/config.py` para ajustar:
- Tamanho de batch
- Taxa de aprendizado
- Número de épocas
- Diretórios de dados
- Parâmetros dos modelos

## 📈 Resultados

Os resultados do treinamento são salvos em:
- `checkpoints/`: Pesos dos modelos treinados
- `results/`: Métricas e visualizações

## 🔧 Requisitos

- Python 3.8+
- PyTorch 2.0+
- CUDA (recomendado para GPU)

## 📝 Licença

Este projeto é fornecido como está para fins educacionais e de pesquisa.
