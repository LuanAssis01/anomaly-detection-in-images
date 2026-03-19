#!/bin/bash

# Script de início rápido para o laboratório
# ResNet-50 e DINOv2 - Com e Sem Dados Sintéticos

echo "======================================================================"
echo "LABORATÓRIO DE DETECÇÃO DE ANOMALIAS EM IMAGENS"
echo "ResNet-50 e DINOv2 - Com e Sem Dados Sintéticos"
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
echo "  --- GERAÇÃO DE DADOS ---"
echo "   1) Gerar dados - cenário SEM sintéticas"
echo "   2) Gerar dados - cenário COM sintéticas"
echo "   3) Gerar dados - ambos os cenários"
echo ""
echo "  --- TREINAMENTO ---"
echo "   4) Treinar tudo (ResNet-50 + DINOv2, ambos cenários)"
echo "   5) Treinar ResNet-50 (ambos cenários)"
echo "   6) Treinar DINOv2 (ambos cenários)"
echo "   7) Treinar tudo - cenário SEM sintéticas"
echo "   8) Treinar tudo - cenário COM sintéticas"
echo ""
echo "  --- AVALIAÇÃO ---"
echo "   9) Avaliar todos os modelos treinados"
echo "  10) Executar análise comparativa"
echo "  11) Demo interativa"
echo ""
echo "  --- PIPELINE COMPLETO ---"
echo "  12) Gerar dados + Treinar + Avaliar (tudo de uma vez)"
echo ""
echo "   0) Sair"
echo ""

read -p "Escolha uma opção [0-12]: " choice

case $choice in
    1)
        echo ""
        echo "Gerando dados - cenário SEM sintéticas..."
        python src/generate_data.py --scenario no_synthetic
        ;;
    2)
        echo ""
        echo "Gerando dados - cenário COM sintéticas..."
        python src/generate_data.py --scenario with_synthetic
        ;;
    3)
        echo ""
        echo "Gerando dados - ambos os cenários..."
        python src/generate_data.py --scenario no_synthetic
        python src/generate_data.py --scenario with_synthetic
        ;;
    4)
        echo ""
        echo "Treinando ResNet-50 + DINOv2 em ambos cenários..."
        python src/train.py --model all --scenario all
        ;;
    5)
        echo ""
        echo "Treinando ResNet-50 em ambos cenários..."
        python src/train.py --model resnet50 --scenario all
        ;;
    6)
        echo ""
        echo "Treinando DINOv2 em ambos cenários..."
        python src/train.py --model dinov2 --scenario all
        ;;
    7)
        echo ""
        echo "Treinando ResNet-50 + DINOv2 - cenário SEM sintéticas..."
        python src/train.py --model all --scenario no_synthetic
        ;;
    8)
        echo ""
        echo "Treinando ResNet-50 + DINOv2 - cenário COM sintéticas..."
        python src/train.py --model all --scenario with_synthetic
        ;;
    9)
        echo ""
        echo "Avaliando todos os modelos..."
        python src/evaluate.py --model all --scenario all --visualize
        ;;
    10)
        echo ""
        echo "Executando análise comparativa..."
        python src/analysis_notebook.py
        ;;
    11)
        echo ""
        echo "Iniciando demo..."
        python src/app.py
        ;;
    12)
        echo ""
        echo "=== PIPELINE COMPLETO ==="
        echo ""
        echo "[1/3] Gerando dados para ambos cenários..."
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
