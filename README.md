## Vita Eustia Process Tool

* A toolkit for translating and repacking PSVita game data of “Aiyoku no Eustia”

-----

![Release](https://img.shields.io/badge/Release-1.0.0-green)
![Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)
![PSV](https://img.shields.io/badge/-PSVita-003791?style=flat&logo=PlayStation)


English • [中文版](README-CN.md)

----

### Showcase
![截图00](screenshot/screenshot00.png)
![截图01](screenshot/screenshot01.png)
![截图02](screenshot/screenshot02.png)
![截图03](screenshot/screenshot03.png)
![截图04](screenshot/screenshot04.png)
![截图05](screenshot/screenshot05.png)
![截图06](screenshot/screenshot06.png)
![截图07](screenshot/screenshot07.png)

-----

## Usage Guide

### Extracting & Repacking .mrg Data Packages
```shell
# Unpack a *.mrg file
# The script automatically detects related .nam & .hed files (three-in-one group; .mrg is required)
python3 mrg_extract_tool.py unpack allpac.mrg

# Repack a folder into a single .mrg file
python3 mrg_extract_tool.py repack --list [dir_name]/list.txt dir_name output.mrg

# Repack a folder into a .mrg + .nam + .hed trio
python3 mrg_extract_tool.py repack --list [dir_name]/list.txt dir_name output.mrg --with-hed
```

### Extracting & Repacking .mzp Texture Packages
```shell
# Unpack a single *.mzp file (supports PNG textures: indexed, 8-bit, 24-bit, 32-bit RGBA)
# The tool creates an “mzp-unpacked” folder containing extracted PNGs and a JSON config file
python3 mzp_extract_tool.py unpack *.mzp

# Unpack all *.mzp files in the target folder
python3 mzp_extract_tool.py unpack [dir_name]

# Repack a single PNG file into an .mzp file
# The tool will automatically locate the matching JSON config file
python3 mzp_extract_tool.py repack *.png

# Repack all PNG files in a folder into their respective .mzp files
python3 mzp_extract_tool.py repack [dir_name]
```

### Processing Script Files (Main Translation Target)
```shell
# 1. Extract allscr.mrg using mrg_extract_tool
#    You will get multiple *.mzx script files
python3 mrg_extract_tool.py unpack allscr.mrg

# 2. Convert *.mzx to *.scr using scr_tool
#    *.scr files are encoded in CP932 and contain original Japanese text and script commands
python3 scr_tool.py unpack [dir_name]

# 3. Translation workflow
# ① Translation table: game_script/translate.csv (Original JP, Translated CN)
# ② Eboot translation table: eboot/translate-info.csv
# ③ Use script_compare.py to generate glyphTable/translate-character.txt (character list)
# ④ Use character_to_glyph_table_tool.py to build glyph PNGs and character mapping (glyphTable/character-mapping.json)
# ⑤ Modify the second column in translation CSVs to adjust Chinese text
# ⑥ Run apply_translate_to_scrs.py to apply translations and generate translated *.scr files
#     (Output: game_script/psv/translated/)
# ⑦ Repack *.scr => *.mzx using scr_tool
python3 scr_tool.py respack [dir_name]

# Finally, repack modified *.mzx files into allscr.mrg
```


### Modifying Hardcoded Text in eboot
```shell
# Apply translations from eboot/translate-info.csv into the eboot binary
# Output: eboot/device/eboot_patched.elf
# After patching, repackage the ELF back into eboot.bin (requires re-signing on hardware)
# See restore_eboot_auth_tool.py for details
python3 modify_eboot_hardcode_text_tool.py
```

### Rebuilding Font Glyphs
```shell
# Automatically generate PNG glyph sheets based on glyphTable/translate-character.txt
# Then repack them into *.mzp files using mzp_extract_tool
# Replace the original .mzp files and rebuild the .mrg font package
python3 character_to_glyph_table_tool.py
```

### Notes

This project is designed specifically for the PSVita version of “Aiyoku no Eustia”.
It does not support other games using the HuneX engine.
The repository is provided solely for educational and localization reference purposes.

For a deeper understanding of the HuneX engine, please refer to:

[【Hintay】PS-HuneX_Tools](https://github.com/Hintay/PS-HuneX_Tools)

[【root-none】PS-HuneX_Tools-CN](https://github.com/lzhhzl/PS-HuneX_Tools-CN)

> This project does not handle script overflow issues in branching dialogue.
For more details, see [root-none’s](https://github.com/lzhhzl) article: [Hunex Engine Localization Notes (3)](https://root-none.pages.dev/#/localization_notes/Hunex_Engine/note_hunex_3).

### Disclaimer & Anti-Piracy Statement
![免责声明&抵制](screenshot/disclaimer.png)

-------

## Special Thanks

[PS-HuneX_Tools](https://github.com/Hintay/PS-HuneX_Tools)

[mahoyo_tools](https://github.com/loicfrance/mahoyo_tools)

[deepLuna](https://github.com/Hakanaou/deepLuna)

[PSV_WHITE_ALBUM2](https://github.com/wetor/PSV_WHITE_ALBUM2)

> Without the hard work of these amazing open-source contributors, this project would not exist.
Huge thanks to all developers for their invaluable contributions!