"""
=============================================================================
  CHANGES TO MAKE IN EXISTING FILES — Real CVR Audio Support
  Only 3 small edits needed. Nothing else changes.
=============================================================================


──────────────────────────────────────────────────────────────────────────────
CHANGE 1 of 3  →  compressor.py  (top of file, in the imports section)
──────────────────────────────────────────────────────────────────────────────

FIND this line (around line 10):
┌──────────────────────────────────────────────────────┐
│  import zstandard as zstd                            │
└──────────────────────────────────────────────────────┘

ADD these lines immediately after it:
┌──────────────────────────────────────────────────────────────────────────────┐
│  # Optional: real CVR audio loading (requires pydub + ffmpeg)               │
│  # If cvr_audio.py is in the same folder, load_cvr_audio() becomes          │
│  # available. If not, simulate_cvr_audio() is used as fallback.             │
│  try:                                                                        │
│      from cvr_audio import load_cvr_audio as _load_real_cvr                 │
│      _REAL_CVR_AVAILABLE = True                                              │
│  except ImportError:                                                         │
│      _REAL_CVR_AVAILABLE = False                                             │
└──────────────────────────────────────────────────────────────────────────────┘


──────────────────────────────────────────────────────────────────────────────
CHANGE 2 of 3  →  pipeline.py  (inside run_airplane(), around line 195)
──────────────────────────────────────────────────────────────────────────────

FIND this block (the CVR generation section):
┌──────────────────────────────────────────────────────────────────────────────┐
│  # ── Generate CVR audio ──────────────────────────────────────────────────  │
│  print(f"[PLANE] Generating {PRE_BUFFER_SEC}s CVR audio...")                │
│  cvr_raw = simulate_cvr_audio(PRE_BUFFER_SEC)                               │
│  print(f"[PLANE] CVR raw: {len(cvr_raw):,} bytes")                         │
└──────────────────────────────────────────────────────────────────────────────┘

REPLACE WITH:
┌──────────────────────────────────────────────────────────────────────────────┐
│  # ── CVR audio: real file or simulation ─────────────────────────────────  │
│  cvr_file = getattr(args, 'cvr_file', None) if 'args' in dir() else None   │
│  if cvr_file and os.path.exists(cvr_file):                                  │
│      from cvr_audio import load_cvr_audio                                   │
│      # last_n_sec=60 keeps only the final 60 seconds                        │
│      # (the critical window right before distress event)                    │
│      cvr_raw = load_cvr_audio(cvr_file, last_n_sec=60)                      │
│  else:                                                                       │
│      print(f"[PLANE] No CVR file provided — using simulated audio")         │
│      cvr_raw = simulate_cvr_audio(60)                                       │
│  print(f"[PLANE] CVR ready: {len(cvr_raw):,} bytes")                       │
└──────────────────────────────────────────────────────────────────────────────┘


──────────────────────────────────────────────────────────────────────────────
CHANGE 3 of 3  →  pipeline.py  (inside main(), the argparse section)
──────────────────────────────────────────────────────────────────────────────

FIND the argparse block (around line 310). It looks like:
┌──────────────────────────────────────────────────────────────────────────────┐
│  parser.add_argument("--flight-id",   default="FLT0001", ...)               │
└──────────────────────────────────────────────────────────────────────────────┘

ADD this line immediately after it:
┌──────────────────────────────────────────────────────────────────────────────┐
│  parser.add_argument("--cvr-file",    default=None,                         │
│                      help="Path to real CVR audio file (.m4a, .mp3, .wav)  │
│                            Last 60 sec will be used. If omitted, uses       │
│                            simulated audio.")                                │
└──────────────────────────────────────────────────────────────────────────────┘


=============================================================================
  HOW TO RUN WITH A REAL m4a FILE
=============================================================================

  # Test the audio file alone (check duration, compression ratio):
  python cvr_audio.py your_recording.m4a 60

  # Full pipeline with real CVR audio:
  python pipeline.py airplane \\
      --peer /ip4/<ip>/udp/8000/quic-v1/p2p/<peer_id> \\
      --cvr-file your_recording.m4a \\
      --distress-at 10 \\
      --flight-id FLT0001

  # Self-test with a real file (no network):
  # (you need to temporarily edit pipeline.py _self_test() to pass a cvr_file)


=============================================================================
  WHAT THE CONVERSION DOES TO YOUR m4a FILE
=============================================================================

  Your m4a  →  typically: 44.1 kHz, stereo, AAC compressed
                           (e.g. 60 sec ≈ 0.5–2 MB depending on bitrate)

  After conversion:
    Step 1  Decode AAC   → raw 44.1 kHz 16-bit stereo PCM
    Step 2  Mix to mono  → 44.1 kHz 16-bit mono
    Step 3  Resample     → 4 kHz 16-bit mono  (4 kHz is enough for voice)
    Step 4  Quantise     → 4 kHz 8-bit mono   (halves size, slight quality loss)
    Result: 4000 bytes/sec  →  240,000 bytes for 60 sec

  After delta + zstd compression:
    Speech/silence audio compresses well (silence = long zero runs)
    Typical CR: 4–6×  →  ~45,000–60,000 bytes for 60 sec
    Tx time @ 7 kbps:  ~0.85 min  (well within budget)


=============================================================================
  IF YOUR m4a IS LONGER THAN 60 SECONDS
=============================================================================

  load_cvr_audio(filepath, last_n_sec=60)
                                 ↑
                          Change this to however many seconds you want.
                          Recommended: 60 (the minute before the event).
                          Max within 10-min budget: ~420 sec (7 min).

  If you don't pass last_n_sec, the ENTIRE file is loaded and compressed.
  Make sure it fits in budget:
    python cvr_audio.py your_file.m4a     ← prints budget estimate


=============================================================================
  INSTALL REQUIREMENTS
=============================================================================

  pip install pydub

  ffmpeg (system-level):
    Ubuntu/Debian:  sudo apt install ffmpeg
    macOS:          brew install ffmpeg
    Windows:        download from https://ffmpeg.org/download.html
                    and add to PATH

"""
