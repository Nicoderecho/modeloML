#!/bin/bash

# ============================================================
# Setup Script - Detector de Trading Sospechoso
# ============================================================
# Ejecutar: bash setup.sh

set -e  # Detener en error

echo "=============================================="
echo "CONFIGURANDO ENTORNO ML - TRADING SOSPECHOSO"
echo "=============================================="
echo ""

# Colores
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 1. Actualizar sistema
echo -e "${YELLOW}[1/6]${NC} Actualizando paquetes del sistema..."
sudo apt update -qq
sudo apt upgrade -y -qq
echo -e "${GREEN}✓${NC} Sistema actualizado"

# 2. Instalar Python y herramientas
echo ""
echo -e "${YELLOW}[2/6]${NC} Instalando Python y herramientas..."
sudo apt install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    curl \
    wget \
    build-essential

python3 --version
echo -e "${GREEN}✓${NC} Python instalado"

# 3. Crear entorno virtual
echo ""
echo -e "${YELLOW}[3/6]${NC} Creando entorno virtual..."
if [ -d "venv" ]; then
    echo "  (venv ya existe, saltando...)"
else
    python3 -m venv venv
    echo -e "${GREEN}✓${NC} Entorno virtual creado en ./venv"
fi

# 4. Activar venv e instalar dependencias
echo ""
echo -e "${YELLOW}[4/6]${NC} Instalando dependencias de Python..."
source venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo -e "${GREEN}✓${NC} Dependencias instaladas"

# 5. Configurar git (si no está configurado)
echo ""
echo -e "${YELLOW}[5/6]${NC} Configurando git..."
if [ -z "$(git config --global user.email 2>/dev/null)" ]; then
    echo "  Introduce tu nombre:"
    read -p "  > " nombre
    echo "  Introduce tu email:"
    read -p "  > " email
    git config --global user.name "$nombre"
    git config --global user.email "$email"
    echo -e "${GREEN}✓${NC} Git configurado"
else
    echo "  (git ya configurado)"
fi

# 6. Agregar helper al bashrc
echo ""
echo -e "${YELLOW}[6/6]${NC} Configurando acceso rápido al entorno..."
BASHRC_LINE="source \"$(pwd)/venv/bin/activate\""
if ! grep -q "$BASHRC_LINE" ~/.bashrc 2>/dev/null; then
    echo "" >> ~/.bashrc
    echo "# Entorno ML Trading (auto-generado)" >> ~/.bashrc
    echo "cd $(pwd)" >> ~/.bashrc
    echo "$BASHRC_LINE" >> ~/.bashrc
    echo -e "${GREEN}✓${NC} Bashrc actualizado"
    echo "  → Ejecuta 'source ~/.bashrc' o reinicia terminal"
else
    echo "  (ya configurado)"
fi

# Verificación final
echo ""
echo "=============================================="
echo -e "${GREEN}SETUP COMPLETADO!${NC}"
echo "=============================================="
echo ""
echo "Para activar el entorno:"
echo "  source venv/bin/activate"
echo ""
echo "Para entrenar el modelo:"
echo "  cd src && python train.py"
echo ""
echo "Para iniciar Jupyter:"
echo "  jupyter notebook"
echo ""
echo "Dependencias instaladas:"
pip list --format=freeze | head -10
