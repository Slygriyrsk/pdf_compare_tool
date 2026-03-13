"""
MODULE 1 — arinc_generator.py
==============================
ARINC 717 telemetry generator: 88 flight parameters, realistic sample rates,
proper 12-bit word packing, sync words, superframe layout.

ARINC 717 Structure (1024 words/frame config):
  Frame     = 4 seconds  = 4 subframes
  Subframe  = 1 second   = 256 words (slots 0–255)
  Word      = 12 bits    (values 0–4095)
  Slot 0    = sync word  (rotates SF1→SF4: 0247, 0132, 0310, 0734 octal)
  Slots 1–255 = parameter data, assigned by sample rate

Parameter sample rates and slot assignment:
  1/sec  → 1 slot, appears in SF1 only (slot repeated every 4 sec)
  2/sec  → 2 slots, SF1+SF3
  4/sec  → 4 slots, one per subframe
  8/sec  → 8 slots, 2 per subframe
  16/sec → 16 slots across subframes

Total: 88 params using 230 of 255 available slots (90.2% utilisation).
"""

import numpy as np
import struct
from dataclasses import dataclass, field
from typing import List, Tuple

# ── ARINC 717 constants ───────────────────────────────────────────────────────
WORDS_PER_FRAME = 1024
SUBFRAMES       = 4
WORDS_PER_SF    = 256          # slots 0–255 per subframe
FRAME_SEC       = 4            # seconds per frame
BITS_PER_WORD   = 12
MAX_12BIT       = 4095
SYNC_WORDS      = [0o0247, 0o0132, 0o0310, 0o0734]   # SF1–SF4

HF_BPS          = 7_000        # 7 kbps link budget
BUFFER_SEC      = 1200         # 20-minute pre/post buffer


# ── Parameter definitions: (name, sample_rate_hz, signal_type, min_val, max_val) ─
# signal_type controls how we synthesize realistic waveforms
PARAM_DEFS = [
    # ── Inertial / Air Data ───────────────────────────────────────────────────
    ("Altitude_ft",        4,  "sinusoidal",    0,    45000),
    ("Airspeed_kts",       4,  "sinusoidal",  100,      550),
    ("Pitch_deg",          8,  "oscillating", -20,       20),
    ("Roll_deg",           8,  "oscillating", -45,       45),
    ("Yaw_deg",            4,  "sinusoidal",    0,      360),
    ("Pitch_rate_dps",     8,  "noise",        -5,        5),
    ("Roll_rate_dps",      8,  "noise",        -5,        5),
    ("Yaw_rate_dps",       4,  "noise",        -3,        3),
    ("Vertical_speed_fpm", 4,  "oscillating",-2000,    2000),
    ("Heading_true_deg",   4,  "ramp",          0,      360),
    ("Heading_mag_deg",    1,  "ramp",          0,      360),
    ("Ground_speed_kts",   4,  "sinusoidal",    0,      550),
    ("Track_angle_deg",    2,  "ramp",          0,      360),
    ("Latitude_deg",       2,  "ramp",        -90,       90),
    ("Longitude_deg",      2,  "ramp",       -180,      180),
    ("GPS_alt_ft",         1,  "sinusoidal",    0,    45000),
    # ── Engine (2 engines) ───────────────────────────────────────────────────
    ("Eng1_N1_pct",        4,  "steady",       70,      100),
    ("Eng1_N2_pct",        4,  "steady",       72,      100),
    ("Eng2_N1_pct",        4,  "steady",       70,      100),
    ("Eng2_N2_pct",        4,  "steady",       72,      100),
    ("Eng1_EGT_degC",      2,  "steady",      400,      900),
    ("Eng2_EGT_degC",      2,  "steady",      400,      900),
    ("Eng1_FF_kgh",        2,  "steady",     1500,     5000),
    ("Eng2_FF_kgh",        2,  "steady",     1500,     5000),
    ("Eng1_OilPress_psi",  1,  "steady",       40,       80),
    ("Eng2_OilPress_psi",  1,  "steady",       40,       80),
    ("Eng1_OilTemp_degC",  1,  "steady",       80,      150),
    ("Eng2_OilTemp_degC",  1,  "steady",       80,      150),
    ("Eng1_Vibration",     2,  "noise",         0,        5),
    ("Eng2_Vibration",     2,  "noise",         0,        5),
    # ── Flight Controls ───────────────────────────────────────────────────────
    ("Flap_pos_deg",       2,  "stepwise",      0,       40),
    ("Slat_pos_deg",       2,  "stepwise",      0,       25),
    ("Gear_pos",           1,  "binary",        0,        1),
    ("Spoiler_pos_deg",    2,  "noise",         0,       60),
    ("Aileron_L_deg",      8,  "oscillating", -20,       20),
    ("Aileron_R_deg",      8,  "oscillating", -20,       20),
    ("Elevator_L_deg",     8,  "oscillating", -25,       25),
    ("Elevator_R_deg",     8,  "oscillating", -25,       25),
    ("Rudder_deg",         8,  "oscillating", -25,       25),
    ("Stab_trim_deg",      2,  "steady",       -5,        5),
    ("AOA_L_deg",          4,  "sinusoidal",   -5,       20),
    ("AOA_R_deg",          4,  "sinusoidal",   -5,       20),
    ("Mach",               4,  "sinusoidal",    0,     0.95),
    # ── Air Data / Environment ────────────────────────────────────────────────
    ("Baro_press_mb",      1,  "steady",      980,     1035),
    ("OAT_degC",           1,  "sinusoidal",  -60,       30),
    ("SAT_degC",           1,  "sinusoidal",  -60,       30),
    ("TAT_degC",           2,  "sinusoidal",  -50,       40),
    # ── Environmental Control ─────────────────────────────────────────────────
    ("Cabin_alt_ft",       1,  "steady",        0,     8000),
    ("Cabin_dp_psi",       1,  "steady",        0,        9),
    ("Cabin_temp_degC",    1,  "steady",       18,       26),
    ("Pack1_flow_kgh",     1,  "steady",      100,      500),
    ("Pack2_flow_kgh",     1,  "steady",      100,      500),
    ("Bleed1_press_psi",   1,  "steady",       20,       50),
    ("Bleed2_press_psi",   1,  "steady",       20,       50),
    # ── Hydraulics ───────────────────────────────────────────────────────────
    ("Hyd1_press_psi",     1,  "steady",     2800,     3100),
    ("Hyd2_press_psi",     1,  "steady",     2800,     3100),
    ("Hyd3_press_psi",     1,  "steady",     2800,     3100),
    # ── Fuel ─────────────────────────────────────────────────────────────────
    ("Fuel_total_kg",      1,  "ramp",       5000,    20000),
    ("Fuel_L_wing_kg",     2,  "ramp",       2000,     8000),
    ("Fuel_R_wing_kg",     2,  "ramp",       2000,     8000),
    ("Fuel_ctr_kg",        1,  "ramp",       1000,     4000),
    # ── Electrical ───────────────────────────────────────────────────────────
    ("AC_bus1_V",          1,  "steady",      110,      120),
    ("AC_bus2_V",          1,  "steady",      110,      120),
    ("DC_bus1_V",          1,  "steady",       27,       29),
    ("DC_bus2_V",          1,  "steady",       27,       29),
    ("Battery_V",          1,  "steady",       24,       28),
    ("APU_N1_pct",         1,  "binary",        0,      100),
    # ── Navigation / Guidance ────────────────────────────────────────────────
    ("ILS_LOC_dev_ddm",    4,  "oscillating", -0.5,     0.5),
    ("ILS_GS_dev_ddm",     4,  "oscillating", -0.5,     0.5),
    ("VOR_dev_deg",        2,  "oscillating", -10,       10),
    ("DME_dist_nm",        1,  "ramp",          0,      400),
    # ── Autopilot / FMS ──────────────────────────────────────────────────────
    ("AP_mode_code",       1,  "binary",        0,        7),
    ("Autothrottle_code",  1,  "binary",        0,        3),
    ("FD_pitch_cmd_deg",   4,  "oscillating", -10,       10),
    ("FD_roll_cmd_deg",    4,  "oscillating", -10,       10),
    # ── Warning / Safety ─────────────────────────────────────────────────────
    ("TCAS_RA_code",       2,  "binary",        0,        7),
    ("GPWS_code",          1,  "binary",        0,        7),
    ("Windshear_flag",     1,  "binary",        0,        1),
    ("Stall_warn",         2,  "binary",        0,        1),
    ("Overspeed_warn",     2,  "binary",        0,        1),
    ("Ground_prox_code",   1,  "binary",        0,        7),
    ("Radio_alt_ft",       4,  "sinusoidal",    0,     2500),
    # ── Landing Gear / Brakes ────────────────────────────────────────────────
    ("Brake_press_L_psi",  2,  "noise",         0,     3000),
    ("Brake_press_R_psi",  2,  "noise",         0,     3000),
    ("Tire_press_L_psi",   1,  "steady",      150,      200),
    ("Tire_press_R_psi",   1,  "steady",      150,      200),
    # ── Discrete flags ───────────────────────────────────────────────────────
    ("Sys_flags_1",        1,  "binary",        0,   0xFFF),
    ("Sys_flags_2",        1,  "binary",        0,   0xFFF),
]

assert len(PARAM_DEFS) == 88, f"Expected 88 params, got {len(PARAM_DEFS)}"


@dataclass
class SlotAssignment:
    """Maps a parameter to which (subframe, slot) positions it occupies."""
    param_idx:  int
    param_name: str
    rate_hz:    int
    # list of (subframe_idx 0-3, slot_idx 1-255) where this param appears
    slots:      List[Tuple[int, int]] = field(default_factory=list)


def build_slot_map() -> List[SlotAssignment]:
    """
    Assign parameter slots according to sample rate.
    Rate 4/sec → 1 slot per subframe (4 slots total, one each SF).
    Rate 8/sec → 2 slots per subframe.
    Rate 2/sec → 1 slot in SF1 and SF3.
    Rate 1/sec → 1 slot in SF1 only.

    Slot 0 is always sync.  We fill slots 1–255 linearly.
    """
    next_slot = [1, 1, 1, 1]   # next free slot per subframe
    assignments = []

    for idx, (name, rate, _, _, _) in enumerate(PARAM_DEFS):
        a = SlotAssignment(idx, name, rate)

        if rate == 1:
            # Once per second → put in SF1 (subframe 0)
            sf, sl = 0, next_slot[0]
            next_slot[0] += 1
            a.slots = [(sf, sl)]

        elif rate == 2:
            # Twice per second → SF1 and SF3
            for sf in [0, 2]:
                a.slots.append((sf, next_slot[sf]))
                next_slot[sf] += 1

        elif rate == 4:
            # Once per subframe
            for sf in range(4):
                a.slots.append((sf, next_slot[sf]))
                next_slot[sf] += 1

        elif rate == 8:
            # Twice per subframe
            for sf in range(4):
                for _ in range(2):
                    a.slots.append((sf, next_slot[sf]))
                    next_slot[sf] += 1

        elif rate == 16:
            for sf in range(4):
                for _ in range(4):
                    a.slots.append((sf, next_slot[sf]))
                    next_slot[sf] += 1

        assignments.append(a)

    used = [next_slot[sf] - 1 for sf in range(4)]
    print(f"[SLOT MAP] Slots used per SF: {used}  (max 255)")
    return assignments


def _synthesize(name, sig_type, lo, hi, t: np.ndarray, rng, idx: int) -> np.ndarray:
    """Generate a physically-plausible waveform for one parameter."""
    N   = len(t)
    mid = (hi + lo) / 2
    rng_val = hi - lo

    if sig_type == "sinusoidal":
        period = 200 + idx * 37          # each param has unique period
        vals = mid + (rng_val / 2) * np.sin(2 * np.pi * t / period) \
               + 0.005 * rng_val * rng.standard_normal(N)

    elif sig_type == "oscillating":
        period = 30 + idx * 11
        vals = 0.35 * rng_val * np.sin(2 * np.pi * t / period + idx) \
               + 0.02 * rng_val * rng.standard_normal(N)

    elif sig_type == "noise":
        vals = np.zeros(N)
        sigma = rng_val * 0.05
        for i in range(1, N):
            vals[i] = vals[i-1] * 0.95 + sigma * rng.standard_normal()

    elif sig_type == "ramp":
        # slowly drifting value (heading, lat, lon, fuel consumption)
        rate  = (rng_val * 0.7) / max(t[-1], 1)
        vals  = lo + (rng_val * 0.15) + rate * t \
                + 0.002 * rng_val * rng.standard_normal(N)

    elif sig_type == "steady":
        # near-constant with small fluctuation
        centre = lo + rng_val * (0.5 + 0.1 * rng.standard_normal())
        vals   = centre + 0.005 * rng_val * rng.standard_normal(N)

    elif sig_type == "stepwise":
        # discrete steps (flaps, slats)
        steps  = np.linspace(lo, hi, 5)
        change = np.cumsum(rng.integers(0, 2, N) * rng.integers(0, 3, N))
        vals   = steps[change % len(steps)]

    elif sig_type == "binary":
        # on/off or discrete codes
        n_states = max(2, int(hi) + 1)
        prob_change = 0.001   # low rate of state change
        state = np.zeros(N, dtype=int)
        cur = int(rng.integers(0, n_states))
        for i in range(N):
            if rng.random() < prob_change:
                cur = int(rng.integers(0, n_states))
            state[i] = cur
        return state.astype(float)

    else:
        vals = np.full(N, mid)

    return np.clip(vals, lo, hi)


def generate_telemetry(duration_sec: int = BUFFER_SEC, seed: int = 42):
    """
    Generate ARINC 717 telemetry for 88 parameters over duration_sec seconds.

    Returns:
      raw_bytes   : 12-bit packed binary (3 bytes per 2 words)
      frame_arr   : np.array shape (duration_sec, 256) — one row per subframe
      slot_map    : list[SlotAssignment] — which slot → which param
      param_data  : dict name→np.array of float values (engineering units)
    """
    print(f"\n{'='*65}")
    print(f"  [GEN] ARINC 717 Telemetry — 88 params, {duration_sec}s "
          f"({duration_sec//4} frames)")
    print(f"{'='*65}")

    rng        = np.random.default_rng(seed)
    slot_map   = build_slot_map()
    total_sfs  = duration_sec          # one subframe per second
    frame_arr  = np.zeros((total_sfs, WORDS_PER_SF), dtype=np.uint16)
    param_data = {}

    # Insert sync words
    for sf_idx in range(total_sfs):
        frame_arr[sf_idx, 0] = SYNC_WORDS[sf_idx % 4]

    # Generate and insert each parameter
    for assign in slot_map:
        idx, name, rate, sig, lo, hi = (
            assign.param_idx, assign.param_name, assign.rate_hz,
            PARAM_DEFS[assign.param_idx][2],
            PARAM_DEFS[assign.param_idx][3],
            PARAM_DEFS[assign.param_idx][4],
        )

        # Sample count = one sample per slot per second
        # For rate=4, param appears 4x per sec, so N = 4 * duration_sec
        N    = rate * duration_sec
        t    = np.linspace(0, duration_sec, N)
        vals = _synthesize(name, sig, lo, hi, t, rng, idx)

        # Quantise to 12-bit
        norm  = (vals - lo) / max(hi - lo, 1e-9)
        words = np.clip((norm * MAX_12BIT).astype(int), 0, MAX_12BIT)

        param_data[name] = vals

        # Place words into frame_arr
        sample_idx = 0
        for sf_idx in range(total_sfs):
            sf_pos = sf_idx % 4    # which of the 4 subframes in a frame
            for (sf_slot, slot) in assign.slots:
                if sf_slot == sf_pos and slot < WORDS_PER_SF:
                    if sample_idx < len(words):
                        frame_arr[sf_idx, slot] = words[sample_idx]
                        sample_idx += 1

    # 12-bit packing: 2 words → 3 bytes
    flat   = frame_arr.reshape(-1)
    packed = bytearray()
    for i in range(0, len(flat) - 1, 2):
        a, b = int(flat[i]), int(flat[i + 1])
        packed.append((a >> 4) & 0xFF)
        packed.append(((a & 0x0F) << 4) | ((b >> 8) & 0x0F))
        packed.append(b & 0xFF)
    if len(flat) % 2:
        a = int(flat[-1])
        packed.append((a >> 4) & 0xFF)
        packed.append((a & 0x0F) << 4)

    raw_bytes = bytes(packed)
    raw_bits  = len(raw_bytes) * 8

    print(f"  Parameters       : {len(PARAM_DEFS)}")
    print(f"  Total subframes  : {total_sfs}  ({total_sfs // 4} frames)")
    print(f"  Total words      : {len(flat):,}")
    print(f"  Raw bytes        : {len(raw_bytes):,}")
    print(f"  Raw Tx @ 7 kbps  : {raw_bits / HF_BPS / 60:.2f} min")

    return raw_bytes, frame_arr, slot_map, param_data


# ── Utility: unpack 12-bit words ─────────────────────────────────────────────
def unpack_words(raw_bytes: bytes) -> list:
    words = []
    for i in range(0, len(raw_bytes) - 2, 3):
        b1, b2, b3 = raw_bytes[i], raw_bytes[i+1], raw_bytes[i+2]
        words.append((b1 << 4) | (b2 >> 4))
        words.append(((b2 & 0x0F) << 8) | b3)
    return words


if __name__ == "__main__":
    raw, frame_arr, slot_map, param_data = generate_telemetry(BUFFER_SEC)
    print("\nFirst 5 slot assignments:")
    for a in slot_map[:5]:
        print(f"  {a.param_name:<30} rate={a.rate_hz}/s  slots={a.slots[:2]}...")
