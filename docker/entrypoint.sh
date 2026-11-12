#!/bin/sh
# Container entrypoint for the Smriti GUI.
#
# Tauri's single-instance plugin and GTK both want a D-Bus session bus, which a
# bare container doesn't have, so start a private one unless the caller passed
# DBUS_SESSION_BUS_ADDRESS in.
set -eu

mkdir -p "$XDG_CONFIG_HOME" "$XDG_DATA_HOME" "$XDG_CACHE_HOME" 2>/dev/null || true

if [ -z "${DISPLAY:-}" ]; then
    echo "smriti: DISPLAY is not set. Start from docker compose (it forwards your X11 display)" >&2
    echo "        and, if the window cannot open, run: xhost +local:docker" >&2
    exit 1
fi

if [ -z "${DBUS_SESSION_BUS_ADDRESS:-}" ] && command -v dbus-run-session >/dev/null 2>&1; then
    exec dbus-run-session -- "$@"
fi
exec "$@"
