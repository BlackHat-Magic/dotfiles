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

Installation with [GNU stow](https://www.gnu.org/software/stow/):

```sh
cd
git clone git@github.com:BlackHat-Magic/dotfiles
cd dotfiles
stow .
```

## Window Manager (Niri)

This config uses [Niri](https://github.com/niri-wm/niri) with the following utilities:
- [brightnessctl](https://github.com/Hummer12007/brightnessctl), [playerctl](https://github.com/altdesktop/playerctl), and [wireplumber](https://gitlab.freedesktop.org/pipewire/wireplumber/) for brightness and media keys and sound.
- [Gnome Polkit](https://gitlab.gnome.org/Archive/policykit-gnome) for authentication
- [grim](https://sr.ht/~emersion/grim/), [swappy](github.com/jtheoof/swappy), and [slurp](https://github.com/emersion/slurp) for screenshots.
- [Mako](https://github.com/emersion/mako) for notifications
- [Nautilus](https://gitlab.gnome.org/GNOME/nautilus) as a file manager.
- [Noto Fonts](https://fonts.google.com/noto) and [Nerd Fonts](https://github.com/ryanoasis/nerd-fonts) for fonts
- [nwg-look](https://github.com/nwg-piotr/nwg-look) for managing themes.
	- [Catppuccin](https://catppuccin.com/) as the main color scheme
	- [cursors](https://catppuccin.com/)
	- [GTK Theme](https://github.com/catppuccin/gtk)
- [polkit-gnome](https://gitlab.gnome.org/Archive/policykit-gnome) for polkit
- [rofi](https://github.com/davatorium/rofi), [rofi-calc](https://github.com/svenstaro/rofi-calc), and [cliphist](https://github.com/sentriz/cliphist) as an application launcher and clipboard manager.
- [swaybg](https://github.com/swaywm/swaybg) for background
- [Waybar](https://github.com/alexays/waybar) for status bar
	- Looking at switching to [ewww](https://github.com/elkowar/eww) or a [Quickshell](https://github.com/alexays/waybar)-based solution
- [wl-clipboard](https://github.com/bugaevc/wl-clipboard) for clipboard history

## Terminal Utilities

This config uses [ghostty](https://github.com/ghostty-org/ghostty) with the following utilities:
- [bottom](https://github.com/clementtsang/bottom) as a file manager.
- [fd](https://github.com/sharkdp/fd) as a `find` replacement.
- [fzf](https://github.com/junegunn/fzf) as a fuzzy finder.
- [kakoune](https://github.com/mawww/kakoune) with [kak-tree-sitter](https://sr.ht/~hadronized/kak-tree-sitter/) and [kakoune-lsp](https://github.com/kakoune-lsp/kakoune-lsp) as a text editor.
- [OpenCode](https://github.com/anomalyco/opencode) for agentic coding.
- [ripgrep](https://github.com/burntsushi/ripgrep) as a `grep` replacement.
- [tldr](https://github.com/zellij-org/zellij) for more readable manpages.
- [wiremix](https://github.com/tsowell/wiremix) for volume control.
- [zellij](https://github.com/zellij-org/zellij) as a terminal multiplexer.
- [zoxide](https://github.com/ajeetdsouza/zoxide) as a `cd` replacement.

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

This repo also contains my configurations for OBS Studio.

### Plugins

It expects these plugins to be installed. `com.obsproject.Studio.Plugin.` is omitted from the start of the flatpak application IDs for brevity.

| **Plugin Name**			| **Flatpak Application ID**	| **Arch/AUR Package**						|
| :---						| :---							| :---										|
| Advanced Masks			| `AdvancedMasks`				| `obs-advanced-masks` (AUR)				|
| Aitum Multistream			| `AitumMultistream`			| `obs-aitum-multistream-bin` (AUR)			|
| Move Transition			| `MoveTransition`				| `obs-move-transition` (AUR) (Out of date) |
| Scale To Sound			| `ScaleToSound`				| `obs-scale-to-sound` (AUR)				|
| Shaderfiler				| `Shaderfilter`				| `obs-shaderfilter-git` (AUR)				|
| Source Record				| `SourceRecord`				| `obs-source-record` (AUR)					|
| Stroke Glow Shadow		| `StrokeGlowShadow`			| `obs-stroke-glow-shadow` (AUR)			|
| Transition Table			| `TransitionTable`				| `obs-transition-table` (AUR)				|
| Wayland Hotkeys			| `WaylandHotkeys`				| `obs-wayland-hotkeys-git` (AUR)			|
| 3D Effect					| `_3DEffect`					| `obs-3d-effect` (AUR)						|
| Waveform					| `waveform`					| N/A										|

## TODO

- [ ] Icons that don't suck

# Acknowledgements

- OBS "Starting Soon," "Be Right Back," and "Ending Stream" screens use a shader based on [Base warp fBM cineshader(https://www.shadertoy.com/view/3sfczf) by TrinketMage on ShaderToy.

## Arch/AUR Packages:

- blueman 2.4.6-2
- brightnessctl 0.5.1-3
- candy-icons-git r1363.b0a85a7-1
- cloc 2.08-1
- dnsmasq 2.92-1
- git 2.53.0-1
- github-cli 2.89.0-1
- less 1:692-1
- nix 2.34.2-1
- ollama 0.18.3-1
