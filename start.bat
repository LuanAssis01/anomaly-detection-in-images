@echo off
chcp 65001 >nul 2>&1

echo ======================================================================
echo LABORATORIO DE DETECCAO DE ANOMALIAS EM IMAGENS
echo ResNet-50 e DINOv2 - 3 Cenarios de Dados
echo ======================================================================
echo.

echo 1. Testando ambiente...
python test.py

echo.
echo 2. Instalando dependencias faltantes (se necessario)...
pip install -q -r requirements.txt

echo.
echo Ambiente configurado!
echo.

echo ======================================================================
echo OPCOES:
echo ======================================================================
echo.
echo   --- GERACAO DE DADOS ---
echo   1) Gerar - SEM augmentation (apenas imagens reais originais)
echo   2) Gerar - COM augmentation, SEM sinteticas
echo   3) Gerar - COM augmentation + imagens sinteticas
echo   4) Gerar - todos os 3 cenarios
echo.
echo   --- TREINAMENTO ---
echo   5) Treinar tudo (ResNet-50 + DINOv2, todos cenarios)
echo   6) Treinar ResNet-50 (todos cenarios)
echo   7) Treinar DINOv2 (todos cenarios)
echo   8) Treinar tudo - cenario SEM augmentation
echo   9) Treinar tudo - cenario COM augmentation, SEM sinteticas
echo  10) Treinar tudo - cenario COM sinteticas
echo.
echo   --- AVALIACAO ---
echo  11) Avaliar todos os modelos treinados
echo  12) Executar analise comparativa
echo  13) Demo interativa
echo.
echo   --- PIPELINE COMPLETO ---
echo  14) Gerar dados + Treinar + Avaliar (tudo de uma vez)
echo.
echo   0) Sair
echo.

set /p choice="Escolha uma opcao [0-14]: "

if "%choice%"=="1" (
    echo.
    echo Gerando dados - SEM augmentation...
    python src/generate_data.py --scenario no_augmentation
) else if "%choice%"=="2" (
    echo.
    echo Gerando dados - COM augmentation, SEM sinteticas...
    python src/generate_data.py --scenario no_synthetic
) else if "%choice%"=="3" (
    echo.
    echo Gerando dados - COM augmentation + sinteticas...
    python src/generate_data.py --scenario with_synthetic
) else if "%choice%"=="4" (
    echo.
    echo Gerando dados - todos os 3 cenarios...
    python src/generate_data.py --scenario no_augmentation
    python src/generate_data.py --scenario no_synthetic
    python src/generate_data.py --scenario with_synthetic
) else if "%choice%"=="5" (
    echo.
    echo Treinando ResNet-50 + DINOv2 em todos os cenarios...
    python src/train.py --model all --scenario all
) else if "%choice%"=="6" (
    echo.
    echo Treinando ResNet-50 em todos os cenarios...
    python src/train.py --model resnet50 --scenario all
) else if "%choice%"=="7" (
    echo.
    echo Treinando DINOv2 em todos os cenarios...
    python src/train.py --model dinov2 --scenario all
) else if "%choice%"=="8" (
    echo.
    echo Treinando - cenario SEM augmentation...
    python src/train.py --model all --scenario no_augmentation
) else if "%choice%"=="9" (
    echo.
    echo Treinando - cenario COM augmentation, SEM sinteticas...
    python src/train.py --model all --scenario no_synthetic
) else if "%choice%"=="10" (
    echo.
    echo Treinando - cenario COM sinteticas...
    python src/train.py --model all --scenario with_synthetic
) else if "%choice%"=="11" (
    echo.
    echo Avaliando todos os modelos...
    python src/evaluate.py --model all --scenario all --visualize
) else if "%choice%"=="12" (
    echo.
    echo Executando analise comparativa...
    python src/analysis_notebook.py
) else if "%choice%"=="13" (
    echo.
    echo Iniciando demo...
    python src/app.py
) else if "%choice%"=="14" (
    echo.
    echo === PIPELINE COMPLETO ===
    echo.
    echo [1/3] Gerando dados para todos os cenarios...
    python src/generate_data.py --scenario no_augmentation
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
