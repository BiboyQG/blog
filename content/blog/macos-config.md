+++
title = 'My macOS Development Environment: A Comprehensive Dotfiles Guide'
date = 2025-08-17T16:53:11-05:00
draft = false
categories = ['Config', 'macOS']
tags = ['Config', 'macOS', 'dotfiles']
+++

After countless hours of tweaking and optimizing, I've finally crafted a macOS development environment that perfectly suits my workflow. This blog post takes you through my complete dotfiles setup, explaining each component and how they work together to create a productive, efficient development experience.

## 🎯 Philosophy

My dotfiles are built around three core principles:
- **Efficiency**: Everything should be accessible with minimal keystrokes
- **Aesthetics**: A beautiful terminal environment that inspires creativity
- **Automation**: Reduce repetitive tasks through smart automation

## 🛠️ Core Tools Overview

Here's the complete arsenal of tools that power my development environment:

| Tool        | Version | Purpose                                                   |
| ----------- | ------- | --------------------------------------------------------- |
| **kitty**       | 0.37.0  | GPU-accelerated terminal emulator with advanced features |
| **zsh**         | Latest  | Modern shell with powerful customization                 |
| **nvim**        | Latest  | Extensible Vim-based text editor (LazyVim config)        |
| **sketchybar**  | Latest  | Highly customizable macOS status bar replacement         |
| **yabai**       | Latest  | Powerful tiling window manager                           |
| **skhd**        | Latest  | Simple hotkey daemon for system shortcuts               |
| **tmux**        | 3.5a    | Terminal multiplexer for session management              |
| **yazi**        | Latest  | Blazing fast terminal file manager                      |
| **lazygit**     | Latest  | Beautiful terminal Git UI                               |
| **hammerspoon** | Latest  | macOS automation framework                               |
| **fastfetch**   | Latest  | System information display tool                          |
| **cloc**        | N/A     | Count lines of code                                      |
| **dust**        | N/A     | A more intuitive version of du in rust                   |
| **asitop**      | N/A     | System monitor for terminal                              |
| **gh**          | N/A     | GitHub CLI                                               |
| **bat**         | N/A     | Cat with wings                                           |
| **oco**         | N/A     | Generate commit messages with LLMs                       |
| **uv**          | N/A     | Rust package manager for Python                          |

## 🚀 Quick Setup

Before diving deep, here's how to set up everything:

> **⚠️ Important**: Disable System Integrity Protection (SIP) first by booting into Recovery Mode and running `csrutil disable`.

```bash
# Install prerequisite tools
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
curl https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh | sh

# Clone and setup dotfiles
git clone git@github.com:BiboyQG/dotfiles.git && cd dotfiles
zsh setup.sh
```

## 🏗️ Architecture Deep Dive

### Shell Configuration (.zshrc)

My Zsh setup is powered by **Zinit** for lightning-fast plugin management and **Powerlevel10k** for a stunning prompt:

#### Key Features:
- **Smart History**: 10,000 entries with deduplication and intelligent search
- **Advanced Completions**: Case-insensitive matching with fuzzy search
- **Essential Plugins**:
  - `zsh-syntax-highlighting` - Real-time syntax highlighting
  - `zsh-autosuggestions` - Intelligent command suggestions
  - `fzf-tab` - Fuzzy finder for tab completions
  - Oh My Zsh snippets for Git, kubectl, and more

#### Power Aliases:
```bash
ll   # eza -alh --icons     - Enhanced file listing
ssh  # kitten ssh          - SSH through Kitty
s    # fastfetch           - System information
l    # lazygit             - Git UI
t    # sudo asitop         - System monitor
y    # yazi function       - File manager with cd support
c    # claude              - Claude Code CLI
```

### Terminal: Kitty Configuration

Kitty serves as my primary terminal with these standout features:

- **Performance**: GPU-accelerated rendering with custom font fallbacks
- **Aesthetics**: 90% background opacity with blur effects
- **Typography**: JetBrains Mono Nerd Font with Monaspace for italics
- **Functionality**:
  - Split layouts and tabs
  - 10,000 lines of scrollback
  - Copy-on-select behavior
  - Custom tab styling with truncated titles

**Key Bindings**:
- `Cmd+1-9`: Quick tab switching
- `Cmd+Enter`: Split current directory
- `F2`: Broadcast mode for multiple panes

### Window Management: Yabai + SKHD

This combination transforms macOS into a tiling window manager powerhouse:

#### Yabai Features:
- **Binary Space Partitioning**: Automatic window tiling
- **Multi-display Support**: Adaptive padding based on screen resolution
- **Smart Focus**: Automatic window focus on space/display changes
- **App Rules**: Specific apps (Raycast, Zoom) excluded from tiling

#### SKHD Shortcuts:
```bash
# Navigation
Alt + h/j/k/l           # Focus windows
Alt + 1-9               # Switch spaces
Shift + Alt + h/j/k/l   # Move windows
Shift + Alt + 1-4       # Move window to space

# Window Management
Alt + f                 # Toggle fullscreen
Alt + z                 # Toggle float
Alt + e                 # Toggle split direction
Alt + n                 # Create new space

# System
Cmd + Return            # Open Kitty terminal
Cmd + Shift + Return    # Open Arc browser
```

### Terminal Multiplexer: Tmux

My tmux configuration focuses on seamless Vim integration:

- **Visual**: Catppuccin theme with transparent background
- **Navigation**: Vim-style pane navigation with `C-h/j/k/l`
- **Splits**: Intuitive `|` and `-` for horizontal/vertical splits
- **Persistence**: Sessions automatically save and restore via tmux-resurrect
- **Performance**: Minimal escape time (10ms) for responsive Vim usage

### Status Bar: Sketchybar

A completely custom status bar written in Lua:

- **System Monitoring**: CPU, memory, network, battery widgets
- **Media Integration**: Now playing information
- **Workspace Display**: Current space and window information
- **Custom Event System**: C-based event providers for real-time updates

### Text Editor: Neovim (LazyVim)

Built on the excellent LazyVim distribution with custom modifications:

- **Plugin Management**: Lazy.nvim for fast, modern plugin management
- **Theme**: Tokyo Night with consistent terminal integration
- **Navigation**: Seamless tmux integration via vim-tmux-navigator
- **Git Integration**: Lazygit integration for version control

### File Management: Yazi

Yazi provides a modern, Rust-based file management experience:

- **Speed**: Lightning-fast directory traversal
- **Multimedia**: Built-in image/video previews
- **Integration**: Custom shell function for directory changing
- **Theme**: Catppuccin Mocha for consistent aesthetics

### Automation: Hammerspoon

Lua-powered macOS automation handles system-level tasks:

- **Proxy Toggle**: `Cmd+P` to toggle ClashX proxy
- **Auto-reply**: `Cmd+G` to toggle WeChat message auto-reply
- **Extensible**: Modular design for easy feature additions

### Development Tools Integration

#### Git Workflow
- **Lazygit**: Beautiful TUI for Git operations
- **GitHub CLI**: Integrated for pull requests and issues
- **Git Config**: Optimized settings for efficient workflows

#### System Monitoring
- **fastfetch**: Instant system information display
- **asitop**: Real-time system resource monitoring
- **dust**: Intuitive disk usage analysis

## 🔧 System Optimizations

The setup script applies numerous macOS tweaks for developers:

### Keyboard & Input
- Blazing fast key repeat (1ms delay, 20ms initial)
- Disabled automatic capitalization, smart quotes, and period substitution
- Three-finger drag enabled

### Finder Enhancements
- Show all file extensions and hidden files
- Path bar and status bar enabled
- List view as default
- Folders sorted first

### Performance Optimizations
- Sleep image file minimized to save disk space
- .DS_Store creation disabled on network/USB volumes
- Dock auto-hide enabled
- Startup sound disabled

## 🎨 Aesthetic Consistency

The entire setup maintains visual consistency through:

- **Color Scheme**: Catppuccin Mocha across all applications
- **Typography**: Nerd Font integration with proper fallbacks
- **Transparency**: Consistent opacity and blur effects
- **Spacing**: Unified padding and margin values

## 🚦 Getting Started

1. **Prerequisites**: Ensure SIP is disabled and essential tools are installed
2. **Installation**: Run the automated setup script
3. **Post-setup**: Configure yabai sudo permissions and start services
4. **Customization**: Adjust themes and keybindings to your preferences

## 💡 Pro Tips

- Use `Cmd+P` frequently to toggle proxy for development
- Leverage the `y` alias for yazi to navigate and change directories efficiently
- Master the yabai shortcuts for lightning-fast window management
- Utilize tmux sessions for project separation
- Keep your dotfiles in version control for easy syncing across machines

## 🔮 Future Enhancements

- Integration with more development tools
- Enhanced automation scripts
- Custom Sketchybar widgets for project-specific information
- Advanced Hammerspoon automations for workflow optimization

## 📝 Conclusion

This dotfiles setup represents years of refinement, creating an environment where everything works in harmony. The combination of modern tools, thoughtful configuration, and aesthetic consistency results in a development environment that's both powerful and enjoyable to use.

Whether you're a seasoned developer looking to optimize your workflow or someone just starting their customization journey, I hope this setup inspires you to create your own perfect development environment.

---

*Want to explore the configuration files in detail? Check out the [dotfiles repository](https://github.com/BiboyQG/dotfiles) for the complete source code and detailed documentation.*
