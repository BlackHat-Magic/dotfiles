#!/bin/bash

if [ "$EUID" -ne 0 ]; then
	exec doas "$0" "$@" || exec sudo "$0" "$@" || { printf "Could not escalate privileges.\n"; exit 1; }
fi
command -v pacman >/dev/null 2>&1 || { printf "No pacman found.\n" >&2; exit 1; }
pacman -Syy --noconfirm archlinux-keyring || { printf "Failed to sync keyring.\n" >&2; exit 1; }
pacman -Syu --noconfirm || { printf "System update failed.\n" >&2; exit 1; }
command -v useradd >/dev/null 2>&1 || pacman -S --noconfirm sed shadow which reflector || \
	{ printf "Failed to install sed, shadow, and which.\n" >&2; exit 1; }

cp /etc/pacman.d/mirrorlist /etc/pacman.d/mirrorlist.bak
reflector --latest 20 \
	--sort rate \
	--protocol https \
	--country United States \
	--save /etc/pacman.d/mirrorlist

while true; do
    read -p "Enter new user username: " username
    if id "$username" >/dev/null 2>&1; then
        printf "User %s already exists. Enter another username.\n" "$username"
        continue
    fi
    break
done
printf "Creating user %s...\n" "$username"
useradd -m -G wheel -s /bin/bash "$username" || { printf "Failed to create user %s.\n" "$username" >&2; exit 1; }

while true; do
    read -s -p "Enter password for $username: " password
    printf "\n"
    read -s -p "Confirm new password: " confirm
    printf "\n"
    if [ "$password" != "$confirm" ]; then
        printf "Passwords do not match.\n"
        continue
    fi
    break
done
printf "Setting password for %s...\n" "$username"
printf "%s:%s" "$username" "$password" | chpasswd

if command -v sudo >/dev/null 2>&1; then
    while true; do
        read -p "Replace sudo with doas? [Y/n] " yn
        case "${yn:-Y}" in
            [Yy]* )
                pacman -Rns --noconfirm sudo || { printf "Failed to remove sudo.\n" >&2; exit 1; }
                pacman -S --noconfirm doas || { printf "Failed to install doas.\n" >&2; exit 1; }
                printf "permit :wheel\n" > /etc/doas.conf || { printf "Failed to write /etc/doas.conf.\n" >&2; exit 1; }
                break;;
            [Nn]* )
                sed -i 's/^# %wheel ALL=(ALL:ALL) ALL/%wheel ALL=(ALL:ALL) ALL/' /etc/sudoers || \
                    { printf "Failed to edit /etc/sudoers.\n" >&2; exit 1; }
                break;;
            * ) printf "Please answer yes or no.\n";;
        esac
    done
elif ! command -v doas >/dev/null 2>&1; then
    pacman -S --noconfirm doas || { printf "Failed to install doas.\n" >&2; exit 1; }
    printf "permit :wheel\n" > /etc/doas.conf || { printf "Failed to write /etc/doas.conf.\n" >&2; exit 1; }
fi

if ! command -v just >/dev/null 2>&1; then
    while true; do
        read -p "Install just? [Y/n] " yn
        case "${yn:-Y}" in
            [Yy]* ) pacman -S --noconfirm just || { printf "Failed to install just.\n" >&2; exit 1; }; break;;
            [Nn]* ) break;;
            * ) printf "Please answer yes or no.\n";;
        esac
    done
fi

