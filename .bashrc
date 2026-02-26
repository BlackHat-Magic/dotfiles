#
# ~/.bashrc
#

# If not running interactively, don't do anything
[[ $- != *i* ]] && return

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

alias ls='ls --color=auto'
export PATH=$PATH:~/.local/bin
export EDITOR=nvim
alias btw='neofetch'
alias htop='bpytop'
alias top='bpytop'
alias py='python'

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
export PATH=/home/blackhatmagic/.opencode/bin:$PATH

export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"  # This loads nvm
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"  # This loads nvm bash_completion

# pnpm
export PNPM_HOME="/home/blackhatmagic/.local/share/pnpm"
case ":$PATH:" in
  *":$PNPM_HOME:"*) ;;
  *) export PATH="$PNPM_HOME:$PATH" ;;
esac
# pnpm end
