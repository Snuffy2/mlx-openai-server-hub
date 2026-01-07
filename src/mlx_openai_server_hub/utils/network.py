"""Network-related helpers for mlx-openai-server-hub."""

from __future__ import annotations

import socket


def is_port_available(host: str, port: int) -> bool:
    """Return True if a TCP port is free for binding on the given host."""
    try:
        addr_info = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror:
        return False

    for family, socktype, proto, _, sockaddr in addr_info:
        with socket.socket(family, socktype, proto) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.settimeout(0.25)
            try:
                sock.bind(sockaddr)
            except OSError:
                return False
    return True
