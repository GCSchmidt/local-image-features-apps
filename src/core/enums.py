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


class ValidConnection(Enum):
    """
    Indicates whether a connection between a pair of images has been verified with inliers of the homography.
    """
    UNKNOWN = -1
    NOTCONNECTED = 0
    CONNECTED = 1


class HomographyReference(Enum):
    """
    Wether homography should map from/to the base image 
    """
    TOBASE = 0
    FROMBASE = 1