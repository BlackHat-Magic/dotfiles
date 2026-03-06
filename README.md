# Dotfiles

Requires the following arch packages:

```sh
sudo pacman -S bottom brightnessctl cliphist flatpak ghostty hyprland mako nwg-look \
    rofi rofi-calc superfile ttf-nerd-fonts ttf-nerd-fonts-common \
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
posterize <image_name> -o <output_name>		# if dotfiles are installed
uv run posterize							# from the project root if `uv` is installed
uvx --from
```


## TODO

- [ ] Icons that don't suck

# Acknowledgements

- [cliphist](https://github.com/sentriz/cliphist) (`contrib/cliphist-rofi-img` used for clipboard management)
