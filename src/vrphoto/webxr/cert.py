"""Self-signed TLS certificate for the LAN viewer server.

WebXR only works in a *secure context*.  On a home network the practical way to get
one on the Quest is a self-signed HTTPS certificate: the user opens
``https://<pc-ip>:8443/`` once in Quest Browser, accepts the warning and from then on
"Enter VR/AR" works.  (Over USB, ``adb reverse`` + ``http://localhost`` is the
alternative - ``localhost`` is always a secure context.)

The certificate is created with :mod:`cryptography` and persisted as ``cert.pem`` /
``key.pem`` in the certificate directory.  It is re-used as long as it still covers all
current LAN addresses and is not about to expire, because every new certificate means
the user has to click through the browser warning again.  When the PC moves between
networks the new certificate keeps the previously covered addresses too (bounded), so
switching back and forth does not cause a new warning each time.

Secondary addresses in ``172.16.0.0/12`` are treated as *volatile*: on Windows the
Hyper-V "Default Switch", WSL 2 and Docker Desktop adapters get a new random address
from that range on every reboot.  A missing volatile address does not force a new
certificate (the Quest can never reach those adapters anyway); it is only included the
next time the certificate is created for another reason.  The first address in ``ips``
(the primary LAN address shown in the QR code) is always required.
"""

from __future__ import annotations

import datetime as _dt
import ipaddress
import os
import re
import socket
import sys
import threading
from pathlib import Path

from ..core.errors import VRPhotoError
from ..core.log import get_logger

log = get_logger(__name__)

__all__ = [
    "COMMON_NAME",
    "VALIDITY_DAYS",
    "RENEW_BEFORE_DAYS",
    "CertificateError",
    "ensure_cert",
    "cert_info",
    "cert_fingerprint",
]

COMMON_NAME = "VR Photo Studio"
VALIDITY_DAYS = 825
RENEW_BEFORE_DAYS = 30
CERT_FILE = "cert.pem"
KEY_FILE = "key.pem"
_MAX_SAN_IPS = 32  # cap for remembered addresses across network changes
_LOOPBACK = "127.0.0.1"

_lock = threading.Lock()
_HOST_LABEL = re.compile(r"^(?!-)[A-Za-z0-9-]{1,63}(?<!-)$")
# Hyper-V Default Switch / WSL 2 / Docker Desktop: re-randomised on every Windows boot.
_VOLATILE_NET = ipaddress.ip_network("172.16.0.0/12")


class CertificateError(VRPhotoError):
    """The HTTPS certificate could not be created or loaded."""


# --------------------------------------------------------------------------- helpers
def _now() -> _dt.datetime:
    return _dt.datetime.now(_dt.UTC)


def _normalize_ips(ips: list[str] | None) -> list[str]:
    """Parse/deduplicate IP strings (order preserving); invalid entries are skipped."""
    out: list[str] = []
    for raw in ips or []:
        try:
            ip = ipaddress.ip_address(str(raw).strip())
        except ValueError:
            log.warning("Ignoring invalid IP address for certificate: %r", raw)
            continue
        s = str(ip)
        if s not in out:
            out.append(s)
    return out


def _is_volatile(ip: str) -> bool:
    """True for addresses of virtual adapters that change on every reboot (see module doc)."""
    try:
        return ipaddress.ip_address(ip) in _VOLATILE_NET
    except ValueError:
        return False


def _dns_names() -> list[str]:
    """``localhost`` plus the machine host name (and ``<host>.local``) when DNS-safe."""
    names = ["localhost"]
    try:
        host = socket.gethostname().strip().rstrip(".")
    except OSError:
        host = ""
    if host and all(_HOST_LABEL.match(label) for label in host.split(".")) and len(host) <= 253:
        for cand in (host.lower(), f"{host.split('.')[0].lower()}.local"):
            if cand not in names:
                names.append(cand)
    return names


def _cert_paths(cert_dir: Path) -> tuple[Path, Path]:
    return cert_dir / CERT_FILE, cert_dir / KEY_FILE


def _load_existing(cert_path: Path, key_path: Path):
    """Return ``(cert, san_ips)`` if the stored pair is readable and matching, else None."""
    from cryptography import x509
    from cryptography.hazmat.primitives import serialization

    if not (cert_path.is_file() and key_path.is_file()):
        return None
    try:
        cert = x509.load_pem_x509_certificate(cert_path.read_bytes())
        key = serialization.load_pem_private_key(key_path.read_bytes(), password=None)
        pub_cert = cert.public_key().public_bytes(
            serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo
        )
        pub_key = key.public_key().public_bytes(
            serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo
        )
    except Exception as exc:  # unreadable, encrypted, unsupported algorithm, ... -> recreate
        log.info("Existing HTTPS certificate is unreadable (%s); creating a new one", exc)
        return None
    if pub_cert != pub_key:
        log.info("HTTPS certificate and key do not match; creating a new pair")
        return None
    try:
        san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
        san_ips = [str(ip) for ip in san.get_values_for_type(x509.IPAddress)]
    except x509.ExtensionNotFound:
        san_ips = []
    return cert, san_ips


def _write_private(path: Path, data: bytes, *, secret: bool) -> None:
    """Atomically write ``data``; private keys get owner-only permissions (best effort)."""
    # unique per process/thread: another VR Photo Studio process (GUI + CLI) may be
    # writing the same folder; os.replace keeps the final file consistent.
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    if secret and sys.platform != "win32":
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
    else:
        tmp.write_bytes(data)
    try:
        os.replace(tmp, path)
    except OSError:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    if sys.platform != "win32":
        try:
            os.chmod(path, 0o600 if secret else 0o644)
        except OSError:
            pass


def _generate(cert_path: Path, key_path: Path, ips: list[str], dns: list[str]) -> None:
    try:
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID
    except ImportError as exc:  # broken installation / incomplete frozen build
        raise CertificateError(
            "The encryption library needed for the HTTPS certificate is missing.",
            hint="Reinstall VR Photo Studio. Meanwhile the USB connection (adb) still works.",
        ) from exc

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name(
        [
            x509.NameAttribute(NameOID.COMMON_NAME, COMMON_NAME),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, COMMON_NAME),
        ]
    )
    sans: list[x509.GeneralName] = [x509.DNSName(d) for d in dns]
    sans += [x509.IPAddress(ipaddress.ip_address(ip)) for ip in ips]
    now = _now()
    ski = x509.SubjectKeyIdentifier.from_public_key(key.public_key())
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - _dt.timedelta(days=1))  # tolerate clock skew on the headset
        .not_valid_after(now + _dt.timedelta(days=VALIDITY_DAYS))
        .add_extension(x509.SubjectAlternativeName(sans), critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=True,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=False,
                crl_sign=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
        .add_extension(ski, critical=False)
        .add_extension(x509.AuthorityKeyIdentifier.from_issuer_subject_key_identifier(ski), critical=False)
        .sign(key, hashes.SHA256())
    )
    key_pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.TraditionalOpenSSL,
        serialization.NoEncryption(),
    )
    # key first: a crash between the writes leaves a mismatching pair, which is detected
    # (public key comparison) and regenerated on the next call.
    _write_private(key_path, key_pem, secret=True)
    _write_private(cert_path, cert.public_bytes(serialization.Encoding.PEM), secret=False)


# --------------------------------------------------------------------------- public API
def ensure_cert(cert_dir: Path, ips: list[str] | None = None) -> tuple[Path, Path]:
    """Return ``(cert.pem, key.pem)`` in ``cert_dir``, creating/renewing them if needed.

    ``ips`` are the LAN addresses the server is reachable on (``None`` -> detected via
    :func:`vrphoto.webxr.net.lan_ips`).  The certificate is a self-signed RSA-2048
    X.509 leaf (CN "VR Photo Studio", SAN = ``ips`` + 127.0.0.1 + ``localhost`` + host
    name, 825 days, CA:false, serverAuth).  An existing pair is kept when it is readable,
    matches, covers every requested IP and is valid for more than 30 more days;
    otherwise a new pair is written atomically.  When renewing because of new IPs, the
    previously covered IPs are kept (newest first, at most 32).  Secondary addresses in
    172.16.0.0/12 (Hyper-V/WSL/Docker virtual switches that change on every reboot) do
    not by themselves trigger a new certificate - see the module docstring.

    Thread-safe.  Raises :class:`CertificateError` if the files cannot be written.
    """
    cert_dir = Path(cert_dir)
    if ips is None:
        from .net import lan_ips

        ips = lan_ips()
    requested = _normalize_ips(list(ips))
    wanted = _normalize_ips(requested + [_LOOPBACK])
    primary = requested[0] if requested else _LOOPBACK
    cert_path, key_path = _cert_paths(cert_dir)
    with _lock:
        try:
            cert_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise CertificateError(
                f"Cannot create the certificate folder {cert_dir}: {exc}",
                hint="Check that the VR Photo Studio data folder is writable.",
            ) from exc
        existing = _load_existing(cert_path, key_path)
        previous: list[str] = []
        if existing is not None:
            cert, san_ips = existing
            now = _now()
            remaining = cert.not_valid_after_utc - now
            missing = [ip for ip in wanted if ip not in san_ips]
            required = [ip for ip in missing if ip in (primary, _LOOPBACK) or not _is_volatile(ip)]
            not_yet_valid = cert.not_valid_before_utc > now  # PC clock was wrong at creation
            if not required and not not_yet_valid and remaining > _dt.timedelta(days=RENEW_BEFORE_DAYS):
                if missing:
                    log.debug("HTTPS certificate kept; not covering virtual adapter(s) %s", ", ".join(missing))
                return cert_path, key_path
            if required:
                log.info("HTTPS certificate does not cover %s; creating a new one", ", ".join(required))
            elif not_yet_valid:
                log.info("HTTPS certificate is not valid yet (clock change?); creating a new one")
            else:
                log.info("HTTPS certificate expires in %d days; renewing", remaining.days)
            # keep stable old addresses (switching networks back needs no new warning);
            # drop old virtual-adapter addresses, they will never come back
            previous = [ip for ip in san_ips if not _is_volatile(ip)]
        merged = _normalize_ips(wanted + previous)[:_MAX_SAN_IPS]
        try:
            _generate(cert_path, key_path, merged, _dns_names())
        except OSError as exc:
            raise CertificateError(
                f"Cannot write the HTTPS certificate to {cert_dir}: {exc}",
                hint="Check that the VR Photo Studio data folder is writable and not locked by antivirus.",
            ) from exc
        log.info("Created HTTPS certificate for %s", ", ".join(merged))
        return cert_path, key_path


def cert_info(cert_path: Path) -> dict:
    """Summary of a PEM certificate: ``{subject, ips, dns, not_before, not_after, sha256}``.

    Dates are ISO-8601 UTC strings.  Raises :class:`CertificateError` if unreadable.
    """
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes

    try:
        cert = x509.load_pem_x509_certificate(Path(cert_path).read_bytes())
    except (OSError, ValueError) as exc:
        raise CertificateError(f"Cannot read certificate {cert_path}: {exc}") from exc
    try:
        san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
        ips = [str(v) for v in san.get_values_for_type(x509.IPAddress)]
        dns = list(san.get_values_for_type(x509.DNSName))
    except x509.ExtensionNotFound:
        ips, dns = [], []
    return {
        "subject": cert.subject.rfc4514_string(),
        "ips": ips,
        "dns": dns,
        "not_before": cert.not_valid_before_utc.isoformat(),
        "not_after": cert.not_valid_after_utc.isoformat(),
        "sha256": cert.fingerprint(hashes.SHA256()).hex(":").upper(),
    }


def cert_fingerprint(cert_path: Path) -> str:
    """SHA-256 fingerprint ``AA:BB:...`` (what Quest Browser shows in certificate details)."""
    return cert_info(cert_path)["sha256"]
