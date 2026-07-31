"""Shared constants and canonical field order."""

from __future__ import annotations

G_STD = 9.80665
KSC_TO_KN_M2 = 98.0665
KG_M2_TO_KN_M2 = G_STD / 1000.0

FEATURE_COLUMNS = [
    "vy_kn",
    "dy_m",
    "vc_kn",
    "dc_m",
    "vu_kn",
    "du_m",
    "t1_s",
]

TARGET_COLUMNS = [
    "theta_io_g",
    "beta_io",
    "theta_ls_g",
    "beta_ls",
    "theta_cp_g",
    "beta_cp",
]

LIMIT_STATES = ("IO", "LS", "CP")
