from datetime import date

VN30 = [
    "ACB", "BID", "BSR", "CTG", "FPT", "GAS", "GVR", "HDB", "HPG", "LPB",
    "MBB", "MCH", "MSN", "MWG", "SAB", "SHB", "SSB", "SSI", "STB", "TCB",
    "TCX", "VCB", "VHM", "VIB", "VIC", "VJC", "VNM", "VPB", "VPL", "VRE",
]

DEFAULT_START = date(2026, 3, 1)
DEFAULT_END = date(2026, 8, 14)
DEFAULT_TRAIN_WINDOW = 60
DEFAULT_K = 4
DEFAULT_STEP = 5
DEFAULT_CONFIRMATION_STEPS = 2
DEFAULT_CONFIDENCE_THRESHOLD = 0.10

FEATURES = [
    "Return20",
    "Volatility20",
    "Beta60",
    "RS20",
    "VolumeZ20",
    "DistanceVN60",
]
