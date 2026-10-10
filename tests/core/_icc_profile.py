"""Tiny ICC v2 matrix/TRC profile writer for tests (no ICC files ship with the test env)."""

from __future__ import annotations

import struct

# Display P3 primaries adapted to D50 (as in Apple's "Display P3" profile)
P3_D50 = {
    "rXYZ": (0.515121, 0.241196, -0.001053),
    "gXYZ": (0.291977, 0.692245, 0.041885),
    "bXYZ": (0.157104, 0.066574, 0.784073),
}
D50 = (0.9642, 1.0, 0.8249)


def _s15(v: float) -> bytes:
    return struct.pack(">i", int(round(v * 65536)))


def _xyz(v: tuple[float, float, float]) -> bytes:
    return b"XYZ " + b"\0" * 4 + b"".join(_s15(c) for c in v)


def _curv(gamma: float) -> bytes:
    return b"curv" + b"\0" * 4 + struct.pack(">IH", 1, int(round(gamma * 256))) + b"\0\0"


def _desc(text: str) -> bytes:
    a = text.encode("ascii") + b"\0"
    body = b"desc" + b"\0" * 4 + struct.pack(">I", len(a)) + a
    body += struct.pack(">II", 0, 0) + struct.pack(">HB", 0, 0) + b"\0" * 67
    return body + b"\0" * (-len(body) % 4)


def rgb_profile(description: str = "Display P3 (test)", primaries=None, gamma: float = 2.2) -> bytes:
    prim = primaries or P3_D50
    tags = [(b"desc", _desc(description)), (b"wtpt", _xyz(D50))]
    tags += [(k.encode(), _xyz(v)) for k, v in prim.items()]
    curve = _curv(gamma)
    tags += [(b"rTRC", curve), (b"gTRC", curve), (b"bTRC", curve)]
    offset = 128 + 4 + 12 * len(tags)
    table, data = b"", b""
    for sig, body in tags:
        table += sig + struct.pack(">II", offset + len(data), len(body))
        data += body + b"\0" * (-len(body) % 4)
    size = offset + len(data)
    header = struct.pack(">I", size) + b"\0" * 4 + struct.pack(">I", 0x02100000) + b"mntrRGB XYZ "
    header += b"\0" * 12 + b"acsp" + b"\0" * 24 + struct.pack(">I", 0) + b"".join(_s15(c) for c in D50)
    header += b"\0" * (128 - len(header))
    return header + struct.pack(">I", len(tags)) + table + data
