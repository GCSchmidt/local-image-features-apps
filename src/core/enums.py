from enum import Enum, auto


# Panoroma
class MatchCountStrategy(Enum):
    ALL = auto()    # count all raw matches
    GOOD = auto()   # count matches passing ratio test (0.8)
    BEST = auto()   # count only the best match per feature


# plotting
class ConnectivityMatrixMode(Enum):
    RATIO = 0
    COUNT = 1