#!/usr/bin/env bash
# install_toolchains.sh — install Go, Rust, and JDK into .tools/ without root or conda.
#
# Usage:
#   bash scripts/install_toolchains.sh
#
# All binaries land in .tools/bin/ which _resolve_executable() checks automatically.
# Safe to re-run; each toolchain is skipped if already present.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TOOLS_DIR="$REPO_ROOT/.tools"
BIN_DIR="$TOOLS_DIR/bin"
mkdir -p "$BIN_DIR"

# ── helpers ────────────────────────────────────────────────────────────────

link() {
    local src="$1" name="$2"
    ln -sf "$src" "$BIN_DIR/$name"
}

download() {
    local url="$1" dest="$2"
    if command -v curl &>/dev/null; then
        curl -fsSL "$url" -o "$dest"
    elif command -v wget &>/dev/null; then
        wget -q "$url" -O "$dest"
    else
        echo "ERROR: neither curl nor wget found" >&2
        exit 1
    fi
}

detect_arch() {
    case "$(uname -m)" in
        x86_64)  echo "amd64" ;;
        aarch64) echo "arm64" ;;
        *)       echo "amd64" ;;  # HPC nodes are almost always x86_64
    esac
}

# ── Go ─────────────────────────────────────────────────────────────────────
GO_VERSION="1.22.4"
GO_DIR="$TOOLS_DIR/go"

if [ -x "$GO_DIR/bin/go" ]; then
    echo "Go already installed: $($GO_DIR/bin/go version)"
else
    echo "Installing Go $GO_VERSION..."
    ARCH=$(detect_arch)
    TARBALL="$TOOLS_DIR/go.tar.gz"
    download "https://go.dev/dl/go${GO_VERSION}.linux-${ARCH}.tar.gz" "$TARBALL"
    tar -C "$TOOLS_DIR" -xzf "$TARBALL"
    rm "$TARBALL"
    echo "Go installed: $($GO_DIR/bin/go version)"
fi
link "$GO_DIR/bin/go" "go"
link "$GO_DIR/bin/gofmt" "gofmt"

# ── Rust ───────────────────────────────────────────────────────────────────
CARGO_HOME="$TOOLS_DIR/rust"
RUSTUP_HOME="$TOOLS_DIR/rustup"

if [ -x "$CARGO_HOME/bin/cargo" ]; then
    echo "Rust already installed: $($CARGO_HOME/bin/cargo --version)"
else
    echo "Installing Rust (stable, minimal profile)..."
    RUSTUP_INIT="$TOOLS_DIR/rustup-init"
    download "https://sh.rustup.rs" "$RUSTUP_INIT"
    chmod +x "$RUSTUP_INIT"
    CARGO_HOME="$CARGO_HOME" RUSTUP_HOME="$RUSTUP_HOME" \
        "$RUSTUP_INIT" -y --default-toolchain stable --profile minimal --no-modify-path
    rm "$RUSTUP_INIT"
    echo "Rust installed: $($CARGO_HOME/bin/cargo --version)"
fi
link "$CARGO_HOME/bin/cargo" "cargo"
link "$CARGO_HOME/bin/rustc" "rustc"

# ── JDK (Eclipse Temurin 21 LTS) ───────────────────────────────────────────
JDK_DIR="$TOOLS_DIR/jdk"

if [ -x "$JDK_DIR/bin/javac" ]; then
    echo "JDK already installed: $($JDK_DIR/bin/javac -version 2>&1)"
else
    echo "Installing JDK 21 (Eclipse Temurin)..."
    ARCH=$(detect_arch)
    JDK_ARCH="$ARCH"
    [ "$ARCH" = "amd64" ] && JDK_ARCH="x64"
    TARBALL="$TOOLS_DIR/jdk.tar.gz"
    download \
        "https://api.adoptium.net/v3/binary/latest/21/ga/linux/${JDK_ARCH}/jdk/hotspot/normal/eclipse?project=jdk" \
        "$TARBALL"
    mkdir -p "$JDK_DIR"
    tar -C "$JDK_DIR" --strip-components=1 -xzf "$TARBALL"
    rm "$TARBALL"
    echo "JDK installed: $($JDK_DIR/bin/javac -version 2>&1)"
fi
link "$JDK_DIR/bin/javac" "javac"
link "$JDK_DIR/bin/java" "java"

# ── Summary ────────────────────────────────────────────────────────────────
echo ""
echo "Toolchains ready in $BIN_DIR:"
ls -1 "$BIN_DIR"
echo ""
echo "The training loop picks these up automatically via _resolve_executable()."
echo "No PATH changes needed."
