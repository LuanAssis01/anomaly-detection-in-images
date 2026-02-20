#!/bin/bash

# Script de início rápido para o laboratório

echo "======================================================================"
echo "LABORATÓRIO DE DETECÇÃO DE ANOMALIAS EM IMAGENS"
echo "======================================================================"
echo ""

# Cores
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 1. Verificar ambiente
echo -e "${YELLOW}1. Testando ambiente...${NC}"
python test.py

echo ""
echo -e "${YELLOW}2. Instalando dependências faltantes (se necessário)...${NC}"
pip install -q -r requirements.txt

echo ""
echo -e "${GREEN}Ambiente configurado!${NC}"
echo ""

# 2. Menu
echo "======================================================================"
echo "OPÇÕES:"
echo "======================================================================"
echo ""
echo "  1) Treinar todos os modelos (CNN, ViT, CvT, DINOv2)"
echo "  2) Treinar apenas ResNet-50 (rápido)"
echo "  3) Treinar apenas ResNet-101"
echo "  4) Treinar apenas CvT (recomendado)"
echo "  5) Treinar apenas DINOv2"
echo "  6) Avaliar modelos treinados"
echo "  7) Executar análise comparativa"
echo "  8) Demo interativa"
echo "  9) Sair"
echo ""

read -p "Escolha uma opção [1-9]: " choice

case $choice in
    1)
        echo ""
        echo "Treinando todos os modelos..."
        python src/train.py --model all --epochs 30
        ;;
    2)
        echo ""
        echo "Treinando ResNet-50..."
        python src/train.py --model resnet50 --epochs 30
        ;;
    3)
        echo ""
        echo "Treinando ResNet-101..."
        python src/train.py --model resnet101
        ;;
    4)
        echo ""
        echo "Treinando CvT-21..."
        python src/train.py --model cvt21 --epochs 50
        ;;
    5)
        echo ""
        echo "Treinando DINOv2..."
        python src/train.py --model dinov2 --epochs 50
        ;;
    6)
        echo ""
        echo "Avaliando modelos..."
        python src/evaluate.py --model all --visualize
        ;;
    7)
        echo ""
        echo "Executando análise comparativa..."
        python src/analysis_notebook.py
        ;;
    8)
        echo ""
        echo "Iniciando demo..."
        python src/app.py
        ;;
    9)
        echo "Saindo..."
        exit 0
        ;;
    *)
        echo "Opção inválida!"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}Concluído!${NC}"
echo ""
