@echo off
REM Script de início rápido para Windows

echo ======================================================================
echo LABORATÓRIO DE DETECÇÃO DE ANOMALIAS EM IMAGENS
echo ======================================================================
echo.

REM 1. Verificar ambiente
echo 1. Testando ambiente...
python test.py

echo.
echo 2. Instalando dependências faltantes (se necessário)...
pip install -q -r requirements.txt

echo.
echo Ambiente configurado!
echo.

REM 2. Menu
echo ======================================================================
echo OPÇÕES:
echo ======================================================================
echo.
echo   1) Treinar todos os modelos (CNN, ViT, CvT, DINOv2)
echo   2) Treinar apenas CNN (rápido)
echo   3) Treinar apenas CvT (recomendado)
echo   4) Treinar apenas DINOv2
echo   5) Avaliar modelos treinados
echo   6) Executar análise comparativa
echo   7) Demo interativa
echo   8) Sair
echo.

set /p choice="Escolha uma opção [1-8]: "

if "%choice%"=="1" (
    echo.
    echo Treinando todos os modelos...
    python src/train.py --model all --epochs 50
) else if "%choice%"=="2" (
    echo.
    echo Treinando ResNet-50...
    python src/train.py --model resnet50 --epochs 30
) else if "%choice%"=="3" (
    echo.
    echo Treinando CvT-21...
    python src/train.py --model cvt21 --epochs 50
) else if "%choice%"=="4" (
    echo.
    echo Treinando DINOv2...
    python src/train.py --model dinov2 --epochs 50
) else if "%choice%"=="5" (
    echo.
    echo Avaliando modelos...
    python src/evaluate.py --model all --visualize
) else if "%choice%"=="6" (
    echo.
    echo Executando análise comparativa...
    python src/analysis_notebook.py
) else if "%choice%"=="7" (
    echo.
    echo Iniciando demo...
    python src/app.py
) else if "%choice%"=="8" (
    echo Saindo...
    exit /b 0
) else (
    echo Opção inválida!
    exit /b 1
)

echo.
echo Concluído!
echo.
pause
