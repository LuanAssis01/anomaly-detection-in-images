#!/bin/bash

# Script de início rápido — ResNet-50 e DINOv2 | 3 Cenários de Dados

echo "======================================================================"
echo "LABORATÓRIO DE DETECÇÃO DE ANOMALIAS EM IMAGENS"
echo "ResNet-50 e DINOv2 - 3 Cenários de Dados"
echo "======================================================================"
echo ""

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}1. Testando ambiente...${NC}"
python test.py

echo ""
echo -e "${YELLOW}2. Instalando dependências faltantes (se necessário)...${NC}"
pip install -q -r requirements.txt

echo ""
echo -e "${GREEN}Ambiente configurado!${NC}"
echo ""

echo "======================================================================"
echo "OPÇÕES:"
echo "======================================================================"
echo ""
echo "  --- GERAÇÃO DE DADOS ---"
echo "   1) Gerar - SEM augmentation (apenas imagens reais originais)"
echo "   2) Gerar - COM augmentation, SEM sintéticas"
echo "   3) Gerar - COM augmentation + imagens sintéticas"
echo "   4) Gerar - todos os 3 cenários"
echo ""
echo "  --- TREINAMENTO ---"
echo "   5) Treinar tudo (ResNet-50 + DINOv2, todos cenários)"
echo "   6) Treinar ResNet-50 (todos cenários)"
echo "   7) Treinar DINOv2 (todos cenários)"
echo "   8) Treinar tudo - cenário SEM augmentation"
echo "   9) Treinar tudo - cenário COM augmentation, SEM sintéticas"
echo "  10) Treinar tudo - cenário COM sintéticas"
echo ""
echo "  --- HYPERPARAMETER SEARCH ---"
echo "  11) Grid Search - ResNet-50 (todos cenários)"
echo "  12) Grid Search - DINOv2 (todos cenários)"
echo "  13) Randomized Search - ResNet-50 (20 trials, todos cenários)"
echo "  14) Randomized Search - DINOv2 (20 trials, todos cenários)"
echo "  15) Grid Search - todos os modelos e cenários"
echo ""
echo "  --- AVALIAÇÃO ---"
echo "  16) Avaliar todos os modelos treinados"
echo "  17) Executar análise comparativa"
echo "  18) Demo interativa"
echo ""
echo "  --- PIPELINE COMPLETO ---"
echo "  19) Gerar dados + Treinar + Avaliar (tudo de uma vez)"
echo ""
echo "   0) Sair"
echo ""

read -p "Escolha uma opção [0-19]: " choice

case $choice in
    1)
        echo ""
        echo "Gerando dados - SEM augmentation..."
        python src/generate_data.py --scenario no_augmentation
        ;;
    2)
        echo ""
        echo "Gerando dados - COM augmentation, SEM sintéticas..."
        python src/generate_data.py --scenario no_synthetic
        ;;
    3)
        echo ""
        echo "Gerando dados - COM augmentation + sintéticas..."
        python src/generate_data.py --scenario with_synthetic
        ;;
    4)
        echo ""
        echo "Gerando dados - todos os 3 cenários..."
        python src/generate_data.py --scenario no_augmentation
        python src/generate_data.py --scenario no_synthetic
        python src/generate_data.py --scenario with_synthetic
        ;;
    5)
        echo ""
        echo "Treinando ResNet-50 + DINOv2 em todos os cenários..."
        python src/train.py --model all --scenario all
        ;;
    6)
        echo ""
        echo "Treinando ResNet-50 em todos os cenários..."
        python src/train.py --model resnet50 --scenario all
        ;;
    7)
        echo ""
        echo "Treinando DINOv2 em todos os cenários..."
        python src/train.py --model dinov2 --scenario all
        ;;
    8)
        echo ""
        echo "Treinando - cenário SEM augmentation..."
        python src/train.py --model all --scenario no_augmentation
        ;;
    9)
        echo ""
        echo "Treinando - cenário COM augmentation, SEM sintéticas..."
        python src/train.py --model all --scenario no_synthetic
        ;;
    10)
        echo ""
        echo "Treinando - cenário COM sintéticas..."
        python src/train.py --model all --scenario with_synthetic
        ;;
    11)
        echo ""
        echo "Grid Search - ResNet-50 (todos cenários)..."
        python src/hyperparam_search.py --model resnet50 --scenario all --method grid
        ;;
    12)
        echo ""
        echo "Grid Search - DINOv2 (todos cenários)..."
        python src/hyperparam_search.py --model dinov2 --scenario all --method grid
        ;;
    13)
        echo ""
        echo "Randomized Search - ResNet-50 (20 trials, todos cenários)..."
        python src/hyperparam_search.py --model resnet50 --scenario all --method random --n-iter 20
        ;;
    14)
        echo ""
        echo "Randomized Search - DINOv2 (20 trials, todos cenários)..."
        python src/hyperparam_search.py --model dinov2 --scenario all --method random --n-iter 20
        ;;
    15)
        echo ""
        echo "Grid Search - todos os modelos e cenários..."
        python src/hyperparam_search.py --model all --scenario all --method grid
        ;;
    16)
        echo ""
        echo "Avaliando todos os modelos..."
        python src/evaluate.py --model all --scenario all --visualize
        ;;
    17)
        echo ""
        echo "Executando análise comparativa..."
        python src/analysis_notebook.py
        ;;
    18)
        echo ""
        echo "Iniciando demo..."
        python src/app.py
        ;;
    19)
        echo ""
        echo "=== PIPELINE COMPLETO ==="
        echo ""
        echo "[1/3] Gerando dados para todos os cenários..."
        python src/generate_data.py --scenario no_augmentation
        python src/generate_data.py --scenario no_synthetic
        python src/generate_data.py --scenario with_synthetic
        echo ""
        echo "[2/3] Treinando todos os modelos..."
        python src/train.py --model all --scenario all
        echo ""
        echo "[3/3] Avaliando todos os modelos..."
        python src/evaluate.py --model all --scenario all --visualize
        echo ""
        echo "Pipeline completo finalizado!"
        ;;
    0)
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
