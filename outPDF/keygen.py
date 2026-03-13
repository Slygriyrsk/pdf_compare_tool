"""
keygen.py
=========
Generate a persistent Ed25519 keypair for authenticated QUIC sessions.

Run ONCE on the airplane side:
    python keygen.py

Creates two files:
    airplane_private.key   — keep on airplane, NEVER share
    airplane_public.key    — copy to every ground station (USB/email/etc.)

The ground station loads airplane_public.key to verify chunk signatures.
The airplane loads airplane_private.key to sign every chunk.

Ed25519 properties:
    - 32-byte private key, 32-byte public key (64 hex chars each)
    - 64-byte signature per chunk (tiny overhead)
    - Signing: < 0.1 ms    Verification: < 0.2 ms
    - Deterministic — same key always produces same signature for same data
"""

import os
import sys
import nacl.signing
import nacl.encoding

PRIVATE_KEY_FILE = "airplane_private.key"
PUBLIC_KEY_FILE  = "airplane_public.key"


def generate_keypair(force: bool = False):
    """Generate and save keypair. Refuses to overwrite unless force=True."""

    if os.path.exists(PRIVATE_KEY_FILE) and not force:
        print(f"[KEYGEN] ERROR: {PRIVATE_KEY_FILE} already exists.")
        print(f"[KEYGEN] Use 'python keygen.py --force' to overwrite.")
        print(f"[KEYGEN] WARNING: Overwriting invalidates all ground station keys!")
        sys.exit(1)

    # Generate fresh Ed25519 keypair
    sk      = nacl.signing.SigningKey.generate()
    priv_hex = sk.encode(nacl.encoding.HexEncoder).decode()
    pub_hex  = sk.verify_key.encode(nacl.encoding.HexEncoder).decode()

    # Save private key (airplane only)
    with open(PRIVATE_KEY_FILE, "w") as f:
        f.write(priv_hex)
    os.chmod(PRIVATE_KEY_FILE, 0o600)   # read only by owner

    # Save public key (distribute to ground stations)
    with open(PUBLIC_KEY_FILE, "w") as f:
        f.write(pub_hex)

    print(f"\n[KEYGEN] ✓ Keypair generated successfully")
    print(f"[KEYGEN] Private key : {PRIVATE_KEY_FILE}  ({len(priv_hex)} hex chars)")
    print(f"[KEYGEN]   → Keep this on the airplane. NEVER share.")
    print(f"[KEYGEN] Public key  : {PUBLIC_KEY_FILE}  ({len(pub_hex)} hex chars)")
    print(f"[KEYGEN]   → Copy this to every ground station.")
    print(f"\n[KEYGEN] Public key hex:")
    print(f"  {pub_hex}")
    print(f"\n[KEYGEN] Ground station command:")
    print(f"  python pipeline.py ground --pubkey {pub_hex}")


def load_private_key(path: str = PRIVATE_KEY_FILE) -> nacl.signing.SigningKey:
    """Load private key from file. Raises FileNotFoundError if missing."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Private key file not found: {path!r}\n"
            f"Run 'python keygen.py' first."
        )
    hex_str = open(path).read().strip()
    return nacl.signing.SigningKey(bytes.fromhex(hex_str))


def load_public_key(path: str = PUBLIC_KEY_FILE) -> nacl.signing.VerifyKey:
    """Load public key from file for ground station verification."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Public key file not found: {path!r}\n"
            f"Copy it from the airplane side."
        )
    hex_str = open(path).read().strip()
    return nacl.signing.VerifyKey(bytes.fromhex(hex_str))


if __name__ == "__main__":
    force = "--force" in sys.argv
    generate_keypair(force=force)
