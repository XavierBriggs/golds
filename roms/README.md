# ROM Directory

This directory is for storing ROM files for retro games (NES, SNES, Genesis, etc.).

## Important Notes

- **DO NOT commit ROMs to git** - ROM files are gitignored
- ROMs must be obtained legally (dump from cartridges you own)
- Atari environments require `gymnasium[atari]`/`ale-py` and an Atari ROM set (see Gymnasium/AutoROM docs)

## Setup Instructions

### 1. Obtain ROMs Legally

You can legally obtain ROMs by:
- Dumping cartridges you own using a ROM dumper device (Retrode, INLretro, etc.)
- Some digital purchases include ROM files

### 2. Place ROMs Here

Place your ROM files in this directory. Supported formats:
- `.nes` - Nintendo Entertainment System
- `.sfc`, `.smc` - Super Nintendo
- `.md`, `.gen`, `.bin` - Sega Genesis/Mega Drive
- `.gb`, `.gbc`, `.gba` - Game Boy variants

### 3. Import to stable-retro

```bash
# From the project root
golds rom import ./roms

# Or using uv
uv run golds rom import ./roms
```

### 4. Verify Import

```bash
# Check if a specific game is available
golds rom verify SuperMarioBros-Nes

# List all available games
golds rom list
```

## Expected Files for GOLDS Games

### Super Mario Bros (NES)
- Filename: `Super Mario Bros. (Japan, USA).nes` or similar
- Game ID after import: `SuperMarioBros-Nes`

### Super Mario Bros. 2 Japan (Lost Levels)
- Filename: `Super Mario Bros. 2 (Japan).nes` or similar
- Game ID after import: `SuperMarioBros2Japan-Nes`

## Troubleshooting

### ROM not importing?
- Ensure the ROM file has the correct checksum (SHA1)
- Try different ROM versions if available
- Check stable-retro documentation for supported ROM versions

### Game not found after import?
```bash
# List all available games to see if it imported with different name
golds rom list --search mario
```

## Resources

- [stable-retro documentation](https://stable-retro.farama.org/)
- [List of supported games](https://github.com/Farama-Foundation/stable-retro)
