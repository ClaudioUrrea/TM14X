#!/bin/bash
# Script de automatización para subir el repositorio a GitHub
# Autor: Claudio Urrea
# Fecha: Noviembre 2025

echo "=================================================="
echo "  Fatigue-Aware HRC - Script de GitHub"
echo "=================================================="
echo ""

# Colores para la salida
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Función para mensajes de éxito
success() {
    echo -e "${GREEN}✓${NC} $1"
}

# Función para mensajes de advertencia
warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Función para mensajes de error
error() {
    echo -e "${RED}✗${NC} $1"
}

# Verificar si estamos en un directorio Git
if [ ! -d ".git" ]; then
    warning "No se detectó repositorio Git. Inicializando..."
    git init
    success "Repositorio Git inicializado"
fi

# Configurar usuario Git (si no está configurado)
if [ -z "$(git config --global user.name)" ]; then
    echo ""
    read -p "Ingresa tu nombre para Git: " git_name
    git config --global user.name "$git_name"
    success "Nombre de usuario configurado: $git_name"
fi

if [ -z "$(git config --global user.email)" ]; then
    echo ""
    read -p "Ingresa tu email para Git: " git_email
    git config --global user.email "$git_email"
    success "Email configurado: $git_email"
fi

# Organizar archivos
echo ""
echo "Organizando archivos..."

# Crear carpeta visualizations si no existe
if [ ! -d "visualizations" ]; then
    mkdir visualizations
    success "Carpeta visualizations creada"
fi

# Mover imágenes a visualizations
for img in collision_analysis.png fatigue_trajectory.png semaphore_distribution.png skill_vs_fatigue.png; do
    if [ -f "$img" ]; then
        mv "$img" visualizations/
        success "Movido: $img → visualizations/"
    fi
done

# Mostrar archivos a subir
echo ""
echo "Archivos a subir:"
echo "----------------"
ls -lh | grep -v "^d" | awk '{print $9, "(" $5 ")"}'
echo ""
ls -lh visualizations/ | grep -v "^d" | awk '{print "visualizations/" $9, "(" $5 ")"}'

# Preguntar por el nombre de usuario de GitHub
echo ""
read -p "Ingresa tu nombre de usuario de GitHub: " github_username

# Preguntar por el nombre del repositorio
echo ""
read -p "Ingresa el nombre del repositorio (sugerido: fatigue-aware-hrc-digital-twin): " repo_name
repo_name=${repo_name:-fatigue-aware-hrc-digital-twin}

# Actualizar README con el username correcto
echo ""
echo "Actualizando README con tu información..."
sed -i.bak "s/your-username/$github_username/g" README.md
sed -i.bak "s/USERNAME/$github_username/g" GUIA_GITHUB_FIGSHARE.md
rm README.md.bak GUIA_GITHUB_FIGSHARE.md.bak 2>/dev/null
success "README actualizado"

# Añadir archivos
echo ""
echo "Añadiendo archivos al staging area..."
git add .
success "Archivos añadidos"

# Mostrar estado
echo ""
echo "Estado del repositorio:"
git status --short

# Hacer commit
echo ""
echo "Creando commit..."
git commit -m "Initial commit: Fatigue-Aware Task Reallocation Framework

- Add main simulation algorithm (Dynamic_Threshold_Algorithm.py)
- Add simulation data (n=1000 episodes)
- Add statistical validation results
- Add visualizations (collision analysis, fatigue trajectory, etc.)
- Add comprehensive documentation (README, LICENSE, CITATION)
- Results validated: 99.90% collision-free rate
- Statistical significance: Friedman χ²(3)=3000.00, p<0.001"

if [ $? -eq 0 ]; then
    success "Commit creado exitosamente"
else
    error "Error al crear commit"
    exit 1
fi

# Configurar remote
echo ""
echo "Configurando repositorio remoto..."
git remote add origin "https://github.com/$github_username/$repo_name.git" 2>/dev/null
if [ $? -eq 0 ]; then
    success "Remoto configurado: https://github.com/$github_username/$repo_name.git"
else
    warning "El remoto ya existe, omitiendo..."
fi

# Renombrar rama a main
git branch -M main
success "Rama renombrada a 'main'"

# Instrucciones finales
echo ""
echo "=================================================="
echo "  ¡Todo listo para subir a GitHub!"
echo "=================================================="
echo ""
echo "INSTRUCCIONES FINALES:"
echo ""
echo "1. Asegúrate de haber creado el repositorio en GitHub:"
echo "   https://github.com/new"
echo ""
echo "2. Ejecuta el siguiente comando para subir:"
echo "   ${GREEN}git push -u origin main${NC}"
echo ""
echo "3. Si te pide autenticación, usa tu Personal Access Token:"
echo "   - Ve a: Settings → Developer settings → Personal access tokens"
echo "   - Genera un token con permisos 'repo'"
echo "   - Usa el token como contraseña"
echo ""
echo "4. Después de subir, configura:"
echo "   - Topics/Tags en la página del repositorio"
echo "   - Release v1.0.0"
echo "   - GitHub Pages (opcional)"
echo ""
echo "5. Para más detalles, consulta: GUIA_GITHUB_FIGSHARE.md"
echo ""
echo "URL del repositorio: https://github.com/$github_username/$repo_name"
echo ""
echo "=================================================="
echo ""

# Ofrecer subir automáticamente
read -p "¿Quieres subir a GitHub ahora? (s/N): " upload_now
if [[ $upload_now =~ ^[Ss]$ ]]; then
    echo ""
    echo "Subiendo a GitHub..."
    git push -u origin main
    if [ $? -eq 0 ]; then
        success "¡Subida exitosa!"
        echo ""
        echo "Tu repositorio está disponible en:"
        echo "https://github.com/$github_username/$repo_name"
    else
        error "Error al subir. Revisa tu autenticación."
        echo ""
        echo "Intenta subir manualmente con:"
        echo "git push -u origin main"
    fi
else
    echo ""
    warning "No se subió automáticamente."
    echo "Cuando estés listo, ejecuta: git push -u origin main"
fi

echo ""
echo "=================================================="
echo "  Script completado"
echo "=================================================="
