# types.py

import numpy as np
import numpy.typing as npt
import cv2
from typing import TypeAlias

# Macthes
Match: TypeAlias = cv2.DMatch
MatchGroup: TypeAlias = tuple[Match, ...]
KnnMatches: TypeAlias = tuple[MatchGroup, ...]
MatchList: TypeAlias = list[Match]
KnnMatchesList: TypeAlias = list[MatchList]

# Keypoints/Features
KPGroup: TypeAlias = tuple[cv2.KeyPoint, ...]

# Descriptors 
ORBDescriptor: TypeAlias = npt.NDArray[np.uint8]   # (32,)
ORBDescriptors: TypeAlias = npt.NDArray[np.uint8]  # (N, 32)

SIFTDescriptor: TypeAlias = npt.NDArray[np.float32]  # (128,)
SIFTDescriptors: TypeAlias = npt.NDArray[np.float32]  # (N, 128)