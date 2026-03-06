# Dotfiles

Requires the following arch packages:

```sh
sudo pacman -S bottom brightnessctl cliphist fd flatpak fzf ghostty hyprland mako nwg-look \
    rofi rofi-calc superfile ttf-nerd-fonts ttf-nerd-fonts-common zoxide \
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


## TODO

- [ ] Icons that don't suck

# Acknowledgements

- [cliphist](https://github.com/sentriz/cliphist) (`contrib/cliphist-rofi-img` used for clipboard management)
