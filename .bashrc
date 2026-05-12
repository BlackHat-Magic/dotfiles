#
# ~/.bashrc
#

# If not running interactively, don't do anything
[[ $- != *i* ]] && return

source /usr/share/nvm/init-nvm.sh

# Function to get the current Git branch and status
__git_info() {
  local git_dir=$(git rev-parse --git-dir 2>/dev/null)
  if [ -n "$git_dir" ]; then
    local branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null)
    local status=$(git status --porcelain 2>/dev/null)
    local symbol=""
    if [ -n "$status" ]; then
      symbol="●"  # Dirty
    else
      symbol="○"  # Clean
    fi
    echo "($branch $symbol) "
  else
    echo ""
  fi
}

# Set the PS1 variable
PS1='\[\e[38;5;8m\][ \[\e[0;38;5;12m\]\W\[\e[38;5;8m\] ] \[\e[0;38;5;7m\]$(__git_info)\[\e[0;38;5;6m\]\t\[\e[0m\] \[\e[38;5;4m\]\$\[\e[0m\] '

# fzf Catppuccin Mocha theme
export FZF_DEFAULT_OPTS=" \
--color=bg+:#313244,bg:#1E1E2E,spinner:#F5E0DC,hl:#F38BA8 \
--color=fg:#CDD6F4,header:#F38BA8,info:#CBA6F7,pointer:#F5E0DC \
--color=marker:#B4BEFE,fg+:#CDD6F4,prompt:#CBA6F7,hl+:#F38BA8 \
--color=selected-bg:#45475A \
--color=border:#6C7086,label:#CDD6F4"

alias ls='ls --color=auto'
export PATH=$PATH:~/.local/bin
export EDITOR=kak
alias btw='neofetch'
alias cat='bat --theme dark'
alias cd='z'
alias diff='difft'
alias find='fd'
alias grep='rg'
alias htop='btm'
alias py='python'
alias sudo='doas'
alias top='btm'
fk() {
  local file
  if git rev-parse --git-dir > /dev/null 2>&1; then
    file=$(git ls-files | fzf --preview 'bat --color=always {}' --preview-window=right:60%)
  else
    file=$(fd --type f --hidden --exclude node_modules --exclude __pycache__ --exclude .git --exclude .venv --exclude venv --exclude .env --exclude target --exclude dist --exclude build | fzf --preview 'bat --color=always {}' --preview-window=right:60%)
  fi

  if [[ -n "$file" ]]; then
    kak "$file"
  fi
}

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/usr/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/usr/etc/profile.d/conda.sh" ]; then
        . "/usr/etc/profile.d/conda.sh"
    else
        export PATH="/usr/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

# opencode
export PATH=$HOME/.opencode/bin:$PATH

export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"  # This loads nvm
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"  # This loads nvm bash_completion

# pnpm
export PNPM_HOME="$HOME/.local/share/pnpm"
case ":$PATH:" in
  *":$PNPM_HOME:"*) ;;
  *) export PATH="$PNPM_HOME:$PATH" ;;
esac
# pnpm end

eval "$(zoxide init bash)"

