@echo off
chcp 65001 >nul 2>&1
REM Script de início rápido para Windows

echo ======================================================================
echo LABORATORIO DE DETECCAO DE ANOMALIAS EM IMAGENS
echo ResNet-50 e DINOv2 - Com e Sem Dados Sinteticos
echo ======================================================================
echo.

REM 1. Verificar ambiente
echo 1. Testando ambiente...
python test.py

echo.
echo 2. Instalando dependencias faltantes (se necessario)...
pip install -q -r requirements.txt

echo.
echo Ambiente configurado!
echo.

REM 2. Menu
echo ======================================================================
echo OPCOES:
echo ======================================================================
echo.
echo   --- GERACAO DE DADOS ---
echo   1) Gerar dados - cenario SEM sinteticas
echo   2) Gerar dados - cenario COM sinteticas
echo   3) Gerar dados - ambos os cenarios
echo.
echo   --- TREINAMENTO ---
echo   4) Treinar tudo (ResNet-50 + DINOv2, ambos cenarios)
echo   5) Treinar ResNet-50 (ambos cenarios)
echo   6) Treinar DINOv2 (ambos cenarios)
echo   7) Treinar tudo - cenario SEM sinteticas
echo   8) Treinar tudo - cenario COM sinteticas
echo.
echo   --- AVALIACAO ---
echo   9) Avaliar todos os modelos treinados
echo  10) Executar analise comparativa
echo  11) Demo interativa
echo.
echo   --- PIPELINE COMPLETO ---
echo  12) Gerar dados + Treinar + Avaliar (tudo de uma vez)
echo.
echo   0) Sair
echo.

set /p choice="Escolha uma opcao [0-12]: "

if "%choice%"=="1" (
    echo.
    echo Gerando dados - cenario SEM sinteticas...
    python src/generate_data.py --scenario no_synthetic
) else if "%choice%"=="2" (
    echo.
    echo Gerando dados - cenario COM sinteticas...
    python src/generate_data.py --scenario with_synthetic
) else if "%choice%"=="3" (
    echo.
    echo Gerando dados - ambos os cenarios...
    python src/generate_data.py --scenario no_synthetic
    python src/generate_data.py --scenario with_synthetic
) else if "%choice%"=="4" (
    echo.
    echo Treinando ResNet-50 + DINOv2 em ambos cenarios...
    python src/train.py --model all --scenario all
) else if "%choice%"=="5" (
    echo.
    echo Treinando ResNet-50 em ambos cenarios...
    python src/train.py --model resnet50 --scenario all
) else if "%choice%"=="6" (
    echo.
    echo Treinando DINOv2 em ambos cenarios...
    python src/train.py --model dinov2 --scenario all
) else if "%choice%"=="7" (
    echo.
    echo Treinando ResNet-50 + DINOv2 - cenario SEM sinteticas...
    python src/train.py --model all --scenario no_synthetic
) else if "%choice%"=="8" (
    echo.
    echo Treinando ResNet-50 + DINOv2 - cenario COM sinteticas...
    python src/train.py --model all --scenario with_synthetic
) else if "%choice%"=="9" (
    echo.
    echo Avaliando todos os modelos...
    python src/evaluate.py --model all --scenario all --visualize
) else if "%choice%"=="10" (
    echo.
    echo Executando analise comparativa...
    python src/analysis_notebook.py
) else if "%choice%"=="11" (
    echo.
    echo Iniciando demo...
    python src/app.py
) else if "%choice%"=="12" (
    echo.
    echo === PIPELINE COMPLETO ===
    echo.
    echo [1/3] Gerando dados para ambos cenarios...
    python src/generate_data.py --scenario no_synthetic
    python src/generate_data.py --scenario with_synthetic
    echo.
    echo [2/3] Treinando todos os modelos...
    python src/train.py --model all --scenario all
    echo.
    echo [3/3] Avaliando todos os modelos...
    python src/evaluate.py --model all --scenario all --visualize
    echo.
    echo Pipeline completo finalizado!
) else if "%choice%"=="0" (
    echo Saindo...
    exit /b 0
) else (
    echo Opcao invalida!
    exit /b 1
)

echo.
echo Concluido!
echo.
pause
