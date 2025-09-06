from dataclasses import dataclass

@dataclass
class MatchResult:
    img1: str
    img2: str
    n_feats1: int
    n_feats2: int
    n_inliers: int
    percent_in: float