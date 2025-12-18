#!/bin/bash
# ROM Setup Helper Script for GOLDS

set -e

ROM_DIR="${1:-./roms}"

echo "=========================================="
echo "  GOLDS ROM Setup Helper"
echo "=========================================="
echo ""

# Check if uv is available
if command -v uv &> /dev/null; then
    PYTHON="uv run python"
elif command -v python3 &> /dev/null; then
    PYTHON="python3"
else
    PYTHON="python"
fi

echo "Using Python: $PYTHON"
echo ""

# 1. Atari ROMs
echo "[1/4] Atari ROMs"
echo "  Status: Automatically included with gymnasium"
echo "  Action: No setup needed!"
echo ""

# 2. Check ROM directory
echo "[2/4] ROM Directory: $ROM_DIR"
if [ -d "$ROM_DIR" ]; then
    ROM_COUNT=$(find "$ROM_DIR" -type f \( -name "*.nes" -o -name "*.sfc" -o -name "*.smc" -o -name "*.md" -o -name "*.gen" \) 2>/dev/null | wc -l)
    echo "  Status: Directory exists with $ROM_COUNT ROM files"
else
    echo "  Status: Directory does not exist"
    echo "  Creating directory..."
    mkdir -p "$ROM_DIR"
fi
echo ""

# 3. Import ROMs to stable-retro
echo "[3/4] Importing ROMs to stable-retro"
if [ "$ROM_COUNT" -gt 0 ] 2>/dev/null; then
    echo "  Running: $PYTHON -m retro.import $ROM_DIR"
    $PYTHON -m retro.import "$ROM_DIR" || echo "  Warning: Import may have partially failed"
else
    echo "  Skipping: No ROM files found"
    echo ""
    echo "  To import ROMs later:"
    echo "    1. Place ROM files in $ROM_DIR"
    echo "    2. Run: golds rom import $ROM_DIR"
fi
echo ""

# 4. Verify setup
echo "[4/4] Verifying Setup"
echo ""

# Check for Super Mario Bros
echo "  Checking SuperMarioBros-Nes..."
if $PYTHON -c "import retro; exit(0 if 'SuperMarioBros-Nes' in retro.data.list_games() else 1)" 2>/dev/null; then
    echo "    [OK] SuperMarioBros-Nes is available"
else
    echo "    [--] SuperMarioBros-Nes not found (ROM import needed)"
fi

# List available NES games
echo ""
echo "  Available NES games:"
$PYTHON -c "import retro; games = [g for g in retro.data.list_games() if '-Nes' in g]; print('    ' + '\n    '.join(games[:10]) if games else '    None found')" 2>/dev/null || echo "    Unable to list games"

echo ""
echo "=========================================="
echo "  Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  - Train Space Invaders (no ROM needed):"
echo "      golds train game space_invaders --timesteps 1000000"
echo ""
echo "  - Train Super Mario Bros (after ROM import):"
echo "      golds train game super_mario_bros --timesteps 1000000"
echo ""
