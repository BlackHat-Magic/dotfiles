gui_pacman := """
blueman
brightnessctl
cliphist
ghostty
grim
mako
nautilus
noto-fonts
noto-fonts-cjk
noto-fonts-extra
nwg-look
playerctl
polkit-gnome
rofi
rofi-calc
slurp
swappy
swaybg
ttf-nerd-fonts
ttf-nerd-fonts-common
ttf-nerd-fonts-mono
waybar
wiremix
wireplumber
wl-clipboard
"""

tui_pacman := """
bat
bottom
difftastic
docker
docker-buildx
docker-compose
fd
fzf
ghostty-terminfo
git
git-delta
kakoune
kakoune-lsp
ripgrep
superfile
tldr
whois
zellij
zoxide
"""

gui_paru := """
candy-icons-git
catppuccin-cursors-mocha
catppuccin-gtk-theme-mocha
"""

tui_paru := """
kak-tree-sitter
"""

root_cmd := if `[ "$(id -u)" = "0" ] && echo "true" || echo ""` == "true" { "" } else { `if command -v doas >/dev/null 2>&1; then echo "doas"; elif command -v sudo >/dev/null 2>&1; then echo "sudo"; fi` }

_choose options:
    @printf "%s" "{{ options }}" | fzf --multi --bind 'start:select-all' \
    	--bind 'Ctrl-a:select-all' \
    	--bind 'Ctrl-d:deselect-all' \
    	--bind 'Ctrl-t:toggle-all' \
    	--color='bg+:#313244,bg:#1E1E2E,spinner:#F5E0DC,hl:#F38BA8' \
    	--color='fg:#CDD6F4,header:#F38BA8,info:#CBA6F7,pointer:#F5E0DC' \
    	--color='marker:#B4BEFE,fg+:#CDD6F4,prompt:#CBA6F7,hl+:#F38BA8' \
    	--color='selected-bg:#45475A' \
    	--color='border:#6C7086,label:#CDD6F4' \
    	--marker " " --pointer " " --gutter " " --layout=reverse \
    	--disabled --border rounded --prompt "select packages to install"

install:
    @if [ "$(id -u)" != "0" ] && [ -z "{{ root_cmd }}" ]; then \
    	printf "Must be root or have doas or sudo installed.\n"; exit 1; \
    fi

    command -v curl >/dev/null 2>&1 && command -v fzf >/dev/null 2>&1 && command -v grep >/dev/null 2>&1 && \
    	command -v tee >/dev/null 2>&1 && command -v wget >/dev/null 2>&1 || \
    	{{ root_cmd }} pacman -S --needed --noconfirm coreutils fzf wget || \
    	printf "Unable to install coreutils, fzf, grep, and/or wget.\n"

    if [ -f /etc/wgetrc ]; then \
    	if grep -q "^hsts-file" /etc/wgetrc 2>/dev/null; then \
    		{{ root_cmd }} sed -i "s|^hsts-file.*|hsts-file = /var/cache/wget-hsts|" /etc/wgetrc 2>/dev/null || \
    			printf "Unable to edit hsts-file in /etc/wgetrc (non-fatal); continuing...\n"; \
    	else \
    		printf "\nhsts-file = /var/cache/wget-hsts\n" | {{ root_cmd }} tee -a /etc/wgetrc >/dev/null 2>&1 || \
    			printf "Unable to append hsts-file to /etc/wgetrc (non-fatal); continuing...\n"; \
    	fi \
    else \
    	printf "hsts-file = /var/cache/wget-hsts\n" | {{ root_cmd }} tee /etc/wgetrc >/dev/null 2>&1 || \
    		printf "Unable to create /etc/wgetrc (non-fatal); continuing...\n"; \
    fi

    if [ ! -f /var/cache/wget-hsts ]; then \
    	{{ root_cmd }} touch /var/cache/wget-hsts && \
    	{{ root_cmd }} chmod 666 /var/cache/wget-hsts || \
    		printf "Unable to create /var/cache/wget-hsts (non-fatal); continuing...\n"; \
    fi

    rm -f ~/.wgetrc ~/.wget-hsts 2>/dev/null

    @selected="$(just _choose "{{ gui_pacman }}")"; \
    if [ -n "$$selected" ]; then \
    	printf "%s\n" "$$selected" | xargs -r {{ root_cmd }} pacman -S --needed --noconfirm || \
    	{ printf "Failed to install some packages.\n"; exit 1; }; \
    fi

    @selected="$(just _choose "{{ tui_pacman }}")"; \
    if [ -n "$$selected" ]; then \
    	printf "%s\n" "$$selected" | xargs -r {{ root_cmd }} pacman -S --needed --noconfirm || \
    	{ printf "Failed to install some packages.\n"; exit 1; }; \
    fi

    @if ! command -v paru >/dev/null 2>&1; then \
    	printf "Install paru (AUR helper)? [Y/n] "; \
    	read -r yn; \
    	case "$${yn:-Y}" in \
    		[Yy]*) \
    			git clone https://aur.archlinux.org/paru.git /tmp/paru && \
    			(cd /tmp/paru && makepkg -si --noconfirm) && \
    			rm -rf /tmp/paru || \
    			printf "Failed to install paru.\n";; \
    	esac; \
    fi

    @if command -v paru >/dev/null 2>&1; then \
    	selected="$(just _choose "{{ gui_paru }}")"; \
    	if [ -n "$$selected" ]; then \
    		printf "%s\n" "$$selected" | xargs -r {{ root_cmd }} paru -S --needed --noconfirm || \
    		{ printf "Failed to install some AUR packages.\n"; exit 1; }; \
    	fi; \
    fi

    @if command -v paru >/dev/null 2>&1; then \
    	selected="$(just _choose "{{ tui_paru }}")"; \
    	if [ -n "$$selected" ]; then \
    		printf "%s\n" "$$selected" | xargs -r {{ root_cmd }} paru -S --needed --noconfirm || \
    		{ printf "Failed to install some AUR packages.\n"; exit 1; }; \
    	fi; \
    fi

