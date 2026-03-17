# Dotfiles

Requires the following arch packages:

```sh
sudo pacman -S bottom brightnessctl cliphist fd flatpak fzf ghostty grim mako nwg-look \
    rofi rofi-calc slurp superfile swappy ttf-nerd-fonts ttf-nerd-fonts-common zoxide \
    ttf-nerd-fonts-mono waybar
```

Requires the following AUR packages:

```sh
paru -S catppuccin-cursors-mocha catppuccin-gtk-theme-mocha
```

Requires the following flatpak packages:

```sh
flatpak install org.mozilla.firefox
```

## Python Utilities

This repo contains a bunch of small Python CLI utilities that perform simple tasks that I find useful.

### Posterize

A simple Python utility that posterizes a given image and applies any one of a number of dithering algorithms.

```sh

# if dotfles are stowed in home directory
posterize <image_name> -o <output_namei>

# from the project root if `uv` is installed
uv run posterize

# with `uvx` without installing at all
uvx --from git+https://github.com/BlackHat-Magic/dotfiles posterize
```

## OBS Studio

This repo also contains my configurations for OBS Studio. It expects these plugins to be installed. `com.obsproject.Studio.Plugin.` is omitted from the start of the flatpak application IDs for brevity.

| **Plugin Name**			| **Flatpak Application ID**	| **Arch/AUR Package**						|
| :---						| :---							| :---										|
| Advanced Masks			| `AdvancedMasks`				| `obs-advanced-masks` (AUR)				|
| Aitum Multistream			| `AitumMultistream`			| `obs-aitum-multistream-bin` (AUR)			|
| Aitum Stream Suite		| `AitumStreamSuite`			| N/A										|
| Draw						| `Draw`						| `obs-draw` (AUR)							|
| Move Transition			| `MoveTransition`				| `obs-move-transition` (AUR) (Out of date) |
| Scale To Sound			| `ScaleToSound`				| `obs-scale-to-sound` (AUR)				|
| Advanced Scene Switcher	| `SceneSwitcher`				| `obs-advanced-scene-switcher` (AUR)		|
| Shaderfiler				| `Shaderfilter`				| `obs-shaderfilter-git` (AUR)				|
| Source Clone				| `SourceClone`					| `obs-source-clone` (AUR)					|
| Source Record				| `SourceRecord`				| `obs-source-record` (AUR)					|
| Stroke Glow Shadow		| `StrokeGlowShadow`			| `obs-stroke-glow-shadow` (AUR)			|
| Transition Table			| `TransitionTable`				| `obs-transition-table` (AUR)				|
| Wayland Hotkeys			| `WaylandHotkeys`				| `obs-wayland-hotkeys-git` (AUR)			|
| WebSocket Server (Legacy)	| `WebSocket`					| `obs-websocket-bin` (AUR)					|
| 3D Effect					| `_3DEffect`					| `obs-3d-effect` (AUR)						|
| Waveform					| `waveform`					| N/A										|

## TODO

- [ ] Icons that don't suck

# Acknowledgements

- [cliphist](https://github.com/sentriz/cliphist) (`contrib/cliphist-rofi-img` used for clipboard management)
