"""Tests for vrphoto.webxr.cert (self-signed HTTPS certificate)."""

from __future__ import annotations

import datetime as dt
import ipaddress
import os
import socket
import ssl
import sys
import threading
from pathlib import Path

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

from vrphoto.core.errors import VRPhotoError
from vrphoto.webxr import cert as certmod
from vrphoto.webxr.cert import CertificateError, cert_fingerprint, cert_info, ensure_cert


def _load(path: Path) -> x509.Certificate:
    return x509.load_pem_x509_certificate(path.read_bytes())


def _san(c: x509.Certificate) -> x509.SubjectAlternativeName:
    return c.extensions.get_extension_for_class(x509.SubjectAlternativeName).value


def test_creates_pem_pair_with_expected_fields(tmp_path):
    cert_path, key_path = ensure_cert(tmp_path / "certs", ["192.168.1.5", "10.0.0.7"])
    assert cert_path == tmp_path / "certs" / "cert.pem"
    assert key_path == tmp_path / "certs" / "key.pem"
    c = _load(cert_path)
    assert c.subject.get_attributes_for_oid(NameOID.COMMON_NAME)[0].value == "VR Photo Studio"
    assert c.issuer == c.subject  # self-signed

    san = _san(c)
    ips = {str(i) for i in san.get_values_for_type(x509.IPAddress)}
    assert {"192.168.1.5", "10.0.0.7", "127.0.0.1"} <= ips
    assert "localhost" in san.get_values_for_type(x509.DNSName)

    bc = c.extensions.get_extension_for_class(x509.BasicConstraints)
    assert bc.critical and bc.value.ca is False
    ku = c.extensions.get_extension_for_class(x509.KeyUsage).value
    assert ku.digital_signature and ku.key_encipherment and not ku.key_cert_sign
    eku = c.extensions.get_extension_for_class(x509.ExtendedKeyUsage).value
    assert ExtendedKeyUsageOID.SERVER_AUTH in list(eku)

    life = c.not_valid_after_utc - c.not_valid_before_utc
    assert dt.timedelta(days=825) <= life <= dt.timedelta(days=827)
    assert c.not_valid_before_utc <= dt.datetime.now(dt.UTC)

    key = serialization.load_pem_private_key(key_path.read_bytes(), password=None)
    assert isinstance(key, rsa.RSAPrivateKey) and key.key_size == 2048
    assert key.public_key().public_numbers() == c.public_key().public_numbers()


def test_hostname_in_san_when_dns_safe(tmp_path, monkeypatch):
    monkeypatch.setattr(certmod.socket, "gethostname", lambda: "Gaming-PC")
    cert_path, _ = ensure_cert(tmp_path, ["192.168.1.5"])
    dns = _san(_load(cert_path)).get_values_for_type(x509.DNSName)
    assert "gaming-pc" in dns and "gaming-pc.local" in dns


def test_bad_hostname_is_skipped(tmp_path, monkeypatch):
    monkeypatch.setattr(certmod.socket, "gethostname", lambda: "bad_host name!")
    cert_path, _ = ensure_cert(tmp_path, ["192.168.1.5"])
    assert _san(_load(cert_path)).get_values_for_type(x509.DNSName) == ["localhost"]


def test_reused_when_ips_still_covered(tmp_path):
    c1, _ = ensure_cert(tmp_path, ["192.168.1.5", "10.0.0.7"])
    serial = _load(c1).serial_number
    mtime = c1.stat().st_mtime_ns
    c2, _ = ensure_cert(tmp_path, ["10.0.0.7"])  # subset -> still covered
    c3, _ = ensure_cert(tmp_path, ["192.168.1.5", "10.0.0.7", "127.0.0.1"])
    assert _load(c2).serial_number == serial == _load(c3).serial_number
    assert c1.stat().st_mtime_ns == mtime


def test_regenerated_on_new_ip_and_keeps_previous(tmp_path):
    c1, _ = ensure_cert(tmp_path, ["192.168.1.5"])
    serial = _load(c1).serial_number
    c2, _ = ensure_cert(tmp_path, ["192.168.2.20"])
    new = _load(c2)
    assert new.serial_number != serial
    ips = [str(i) for i in _san(new).get_values_for_type(x509.IPAddress)]
    assert ips[0] == "192.168.2.20"  # current first
    assert "192.168.1.5" in ips and "127.0.0.1" in ips  # previous network remembered
    # switching back to the first network needs no new certificate
    c3, _ = ensure_cert(tmp_path, ["192.168.1.5"])
    assert _load(c3).serial_number == new.serial_number


def test_remembered_ips_are_capped(tmp_path):
    for i in range(40):
        ensure_cert(tmp_path, [f"10.1.{i}.1"])
    ips = _san(_load(tmp_path / "cert.pem")).get_values_for_type(x509.IPAddress)
    assert len(ips) <= certmod._MAX_SAN_IPS
    assert ipaddress.ip_address("10.1.39.1") in ips


def _write_short_lived(cert_dir: Path, days_left: int, ips: list[str]) -> int:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "VR Photo Studio")])
    now = dt.datetime.now(dt.UTC)
    c = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - dt.timedelta(days=800))
        .not_valid_after(now + dt.timedelta(days=days_left))
        .add_extension(
            x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address(i)) for i in ips]), critical=False
        )
        .sign(key, hashes.SHA256())
    )
    cert_dir.mkdir(parents=True, exist_ok=True)
    (cert_dir / "key.pem").write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
        )
    )
    (cert_dir / "cert.pem").write_bytes(c.public_bytes(serialization.Encoding.PEM))
    return c.serial_number


def test_renewed_when_expiring_soon(tmp_path):
    old = _write_short_lived(tmp_path, 10, ["192.168.1.5", "127.0.0.1"])
    c, _ = ensure_cert(tmp_path, ["192.168.1.5"])
    new = _load(c)
    assert new.serial_number != old
    assert new.not_valid_after_utc - dt.datetime.now(dt.UTC) > dt.timedelta(days=800)


def test_kept_when_expiry_is_far(tmp_path):
    old = _write_short_lived(tmp_path, 60, ["192.168.1.5", "127.0.0.1"])
    c, _ = ensure_cert(tmp_path, ["192.168.1.5"])
    assert _load(c).serial_number == old


@pytest.mark.parametrize("damage", ["corrupt_cert", "corrupt_key", "missing_key", "mismatched_key"])
def test_regenerated_when_files_are_damaged(tmp_path, damage):
    c, k = ensure_cert(tmp_path, ["192.168.1.5"])
    serial = _load(c).serial_number
    if damage == "corrupt_cert":
        c.write_text("not a certificate", encoding="utf-8")
    elif damage == "corrupt_key":
        k.write_bytes(b"-----BEGIN PRIVATE KEY-----\ngarbage\n-----END PRIVATE KEY-----\n")
    elif damage == "missing_key":
        k.unlink()
    else:
        other = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        k.write_bytes(
            other.private_bytes(
                serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
            )
        )
    c2, k2 = ensure_cert(tmp_path, ["192.168.1.5"])
    new = _load(c2)
    assert new.serial_number != serial
    key = serialization.load_pem_private_key(k2.read_bytes(), password=None)
    assert key.public_key().public_numbers() == new.public_key().public_numbers()


def test_invalid_ips_are_ignored(tmp_path):
    c, _ = ensure_cert(tmp_path, ["not-an-ip", "", "192.168.1.5", "192.168.1.5", "999.1.1.1"])
    ips = [str(i) for i in _san(_load(c)).get_values_for_type(x509.IPAddress)]
    assert ips == ["192.168.1.5", "127.0.0.1"]


def test_ips_none_uses_lan_detection(tmp_path, monkeypatch):
    from vrphoto.webxr import net

    monkeypatch.setattr(net, "lan_ips", lambda: ["172.20.1.9"])
    c, _ = ensure_cert(tmp_path, None)
    ips = [str(i) for i in _san(_load(c)).get_values_for_type(x509.IPAddress)]
    assert "172.20.1.9" in ips


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permissions")
def test_key_permissions_owner_only(tmp_path):
    _, k = ensure_cert(tmp_path, ["192.168.1.5"])
    assert (os.stat(k).st_mode & 0o777) == 0o600
    assert not list(tmp_path.glob("*.tmp"))


def test_unwritable_dir_raises_friendly_error(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x", encoding="utf-8")
    with pytest.raises(CertificateError) as ei:
        ensure_cert(blocker / "certs", ["192.168.1.5"])
    assert isinstance(ei.value, VRPhotoError) and ei.value.hint


def test_concurrent_calls_produce_one_consistent_pair(tmp_path):
    results: list[tuple[Path, Path]] = []
    errors: list[BaseException] = []

    def work() -> None:
        try:
            results.append(ensure_cert(tmp_path, ["192.168.1.5"]))
        except BaseException as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=work) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert not errors and len(results) == 6
    c = _load(tmp_path / "cert.pem")
    key = serialization.load_pem_private_key((tmp_path / "key.pem").read_bytes(), password=None)
    assert key.public_key().public_numbers() == c.public_key().public_numbers()


def test_cert_info_and_fingerprint(tmp_path):
    c, _ = ensure_cert(tmp_path, ["192.168.1.5"])
    info = cert_info(c)
    assert "VR Photo Studio" in info["subject"]
    assert "192.168.1.5" in info["ips"] and "localhost" in info["dns"]
    fp = cert_fingerprint(c)
    assert len(fp.split(":")) == 32 and fp == fp.upper()
    with pytest.raises(CertificateError):
        cert_info(tmp_path / "missing.pem")


def test_strict_tls_client_accepts_certificate(tmp_path):
    """Python 3.13's default context (VERIFY_X509_STRICT) must accept the cert by IP and name."""
    c, k = ensure_cert(tmp_path, ["192.168.1.5"])
    sctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    sctx.load_cert_chain(c, k)
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(2)
    port = srv.getsockname()[1]

    def serve() -> None:
        for _ in range(2):
            conn, _ = srv.accept()
            try:
                with sctx.wrap_socket(conn, server_side=True) as tls:
                    tls.sendall(b"ok")
            except (ssl.SSLError, OSError):
                pass

    t = threading.Thread(target=serve, daemon=True)
    t.start()
    cctx = ssl.create_default_context(cafile=str(c))
    try:
        for name in ("127.0.0.1", "localhost"):
            with socket.create_connection(("127.0.0.1", port), timeout=5) as s:
                with cctx.wrap_socket(s, server_hostname=name) as tls:
                    assert tls.recv(2) == b"ok"
    finally:
        srv.close()
        t.join(5)


# --------------------------------------------------------------------------- review additions
def _ips_of(path: Path) -> list[str]:
    return [str(i) for i in _san(_load(path)).get_values_for_type(x509.IPAddress)]


def test_new_virtual_adapter_address_does_not_force_new_certificate(tmp_path):
    """Hyper-V/WSL/Docker adapters (172.16/12) get a new address on every Windows boot;
    that must not invalidate the certificate the user already accepted on the Quest."""
    c1, _ = ensure_cert(tmp_path, ["192.168.1.5", "172.20.48.1"])
    serial = _load(c1).serial_number
    assert "172.20.48.1" in _ips_of(c1)  # included when the certificate is created
    c2, _ = ensure_cert(tmp_path, ["192.168.1.5", "172.27.112.1", "172.31.0.9"])  # after a reboot
    assert _load(c2).serial_number == serial


def test_primary_address_is_always_required_even_in_172_range(tmp_path):
    c1, _ = ensure_cert(tmp_path, ["192.168.1.5"])
    serial = _load(c1).serial_number
    c2, _ = ensure_cert(tmp_path, ["172.16.5.20", "192.168.1.5"])  # LAN really is 172.16.x
    assert _load(c2).serial_number != serial
    assert _ips_of(c2)[0] == "172.16.5.20"


def test_stale_virtual_addresses_are_not_carried_over(tmp_path):
    ensure_cert(tmp_path, ["192.168.1.5", "172.20.48.1"])
    c2, _ = ensure_cert(tmp_path, ["10.0.0.8", "172.25.0.1"])  # new LAN -> regenerated
    ips = _ips_of(c2)
    assert ips[0] == "10.0.0.8"
    assert "192.168.1.5" in ips  # stable previous network kept
    assert "172.25.0.1" in ips  # current virtual adapter included
    assert "172.20.48.1" not in ips  # old virtual adapter dropped


def test_regenerated_when_not_yet_valid(tmp_path):
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "VR Photo Studio")])
    now = dt.datetime.now(dt.UTC)
    future = (
        x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now + dt.timedelta(days=30))  # PC clock was ahead when it was created
        .not_valid_after(now + dt.timedelta(days=400))
        .add_extension(x509.SubjectAlternativeName(
            [x509.IPAddress(ipaddress.ip_address(i)) for i in ("192.168.1.5", "127.0.0.1")]), critical=False)
        .sign(key, hashes.SHA256())
    )
    (tmp_path / "key.pem").write_bytes(key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    (tmp_path / "cert.pem").write_bytes(future.public_bytes(serialization.Encoding.PEM))
    c, _ = ensure_cert(tmp_path, ["192.168.1.5"])
    new = _load(c)
    assert new.serial_number != future.serial_number
    assert new.not_valid_before_utc <= dt.datetime.now(dt.UTC)


def test_encrypted_or_unsupported_key_is_replaced(tmp_path):
    c, k = ensure_cert(tmp_path, ["192.168.1.5"])
    serial = _load(c).serial_number
    key = serialization.load_pem_private_key(k.read_bytes(), password=None)
    k.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                    serialization.BestAvailableEncryption(b"pw")))
    c2, k2 = ensure_cert(tmp_path, ["192.168.1.5"])
    assert _load(c2).serial_number != serial
    serialization.load_pem_private_key(k2.read_bytes(), password=None)


def test_no_temp_files_left_and_failed_replace_cleans_up(tmp_path, monkeypatch):
    ensure_cert(tmp_path, ["192.168.1.5"])
    assert sorted(p.name for p in tmp_path.iterdir()) == ["cert.pem", "key.pem"]

    def failing_replace(src, dst):
        raise PermissionError(13, "locked by antivirus")

    monkeypatch.setattr(certmod.os, "replace", failing_replace)
    with pytest.raises(CertificateError) as ei:
        ensure_cert(tmp_path, ["10.9.9.9"])
    assert ei.value.hint
    assert sorted(p.name for p in tmp_path.iterdir()) == ["cert.pem", "key.pem"]


def test_missing_crypto_library_gives_friendly_error(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "cryptography.x509.oid", None)  # import -> ImportError
    with pytest.raises(CertificateError) as ei:
        ensure_cert(tmp_path / "c", ["192.168.1.5"])
    assert isinstance(ei.value, VRPhotoError) and ei.value.hint
