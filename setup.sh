#!/bin/bash
REPO_DIR=$(dirname $(realpath $0))

echo "=== Setup Tiago_CrowdAware ==="

# 1. Crea venv
echo "Creazione venv..."
python3 -m venv "$REPO_DIR/venv"
"$REPO_DIR/venv/bin/pip" install -r \
  "$REPO_DIR/robot/controllers/crowd-aware/requirements.txt"

# 2. Aggiorna runtime.ini con percorso assoluto corretto
echo "Configurazione runtime.ini..."
cat > "$REPO_DIR/robot/controllers/crowd-aware/runtime.ini" << INIEOF
[python]
COMMAND = $REPO_DIR/venv/bin/python
INIEOF

# 3. Fix percorsi proto Webots
echo "Fix percorsi PROTO..."
WEBOTS_PROJECTS=$(find /usr /home -name "Pedestrian.proto" 2>/dev/null | \
  sed 's|/humans/pedestrian/protos/Pedestrian.proto||' | head -1)

if [ -z "$WEBOTS_PROJECTS" ]; then
  echo "ERRORE: Webots non trovato!"
  exit 1
fi

echo "Webots trovato in: $WEBOTS_PROJECTS"
find "$WEBOTS_PROJECTS" -name "*.proto" -exec \
  sed -i "s|webots://projects/|$WEBOTS_PROJECTS/|g" {} \;

# 4. Symlink texture
echo "Symlink texture HDR..."
sudo mkdir -p /usr/local/webots/projects/default/worlds/textures
sudo ln -sf "$WEBOTS_PROJECTS/default/worlds/textures/cubic" \
  /usr/local/webots/projects/default/worlds/textures/cubic

echo "=== Setup completato! ==="
echo "Ora puoi aprire robot/worlds/tiago.wbt in Webots"
