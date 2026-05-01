import pandas as pd
import numpy as np
from classes.orb_image import ORBImage
from core.types import KnnMatchesList
from image_stitching import pipeline
from core.enums import MatchCountStrategy as MCS


class PanoMatchingDF:

    def __init__(self, orb_images: list[ORBImage], matches: KnnMatchesList):
        self.matching_df = pd.DataFrame()
        self._image_names = [oi.filename for oi in orb_images]
        self._kp_ranges: np.ndarray
        self.generate_df(orb_images, matches)
    
    @property
    def n_images(self):
        return len(self._image_names)
    @property
    def image_names(self):
        return self._image_names

    def generate_df(self, orb_images: list[ORBImage], matches: KnnMatchesList):
        self._kp_ranges = pipeline.get_kps_ranges(orb_images)
        img_id_col = []
        matched_id_col = []
        matched_img_col = []
        distance_col = []
        
        for id, ml in enumerate(matches):
            distance_entry = []
            matched_id_entry = []
            matched_img_entry = []
            img_id_entry = pipeline.get_img_id_from_kp_id(id, self._kp_ranges)
            for m in ml:
                distance = m.distance
                matched_id = m.trainIdx
                matched_img_id = pipeline.get_img_id_from_kp_id(matched_id, self._kp_ranges)
                distance_entry.append(distance)
                matched_id_entry.append(matched_id)
                matched_img_entry.append(matched_img_id)
            
            img_id_col.append(img_id_entry)
            matched_id_col.append(matched_id_entry)
            matched_img_col.append(matched_img_entry)
            distance_col.append(distance_entry)

        self.matching_df['ImgId'] = img_id_col
        self.matching_df['Matched_Ids'] = matched_id_col
        self.matching_df['Matched_ImgIds'] = matched_img_col
        self.matching_df['Distances'] = distance_col

    def get_number_of_features_in_img(self, id) -> int:
        """
        Gets the total number of features in image with id.
        Returns:
            int: total features
        """
        total = self._kp_ranges[id+1] - self._kp_ranges[id] 
        return total

    def get_number_of_matches_per_image_pair(self, strategy=MCS.GOOD) -> pd.DataFrame:
        """
        Generate a dataframe specifying the amount of matches between all image pairs.
        Based on MatchCountStrategy (All, Good, Best)
        Returns:
            pd.DataFrame: collumns: ImgId (reference image id),
            MatchedWith (target image id),
            Count (number of matches)
        """
        if strategy == MCS.ALL:
            temp_df = self.matching_df.explode('Matched_ImgIds')
            result = (
                temp_df
                .groupby(['ImgId', 'Matched_ImgIds'])
                .size()
                .reset_index(name='Count')
                .rename(columns={'Matched_ImgIds': 'MatchedWith'})
            )
        elif strategy == MCS.GOOD:
            temp_df = self.matching_df[
                self.matching_df['Distances'].apply(
                    lambda d: len(d) >= 2 and d[0] < 0.8 * d[1]
                )
            ]
            temp_df = temp_df.explode('Matched_ImgIds')
            result = (
                temp_df
                .groupby(['ImgId', 'Matched_ImgIds'])
                .size()
                .reset_index(name='Count')
                .rename(columns={'Matched_ImgIds': 'MatchedWith'})
            )
        elif strategy == MCS.BEST:
            temp_df = self.matching_df[
                self.matching_df['Distances'].apply(lambda d: len(d) >= 1)
            ].copy()
            temp_df['Matched_ImgIds'] = temp_df.apply(
                lambda row: [row['Matched_ImgIds'][0]], axis=1
            )
            temp_df = temp_df.explode('Matched_ImgIds')
            result = (
                temp_df
                .groupby(['ImgId', 'Matched_ImgIds'])
                .size()
                .reset_index(name='Count')
                .rename(columns={'Matched_ImgIds': 'MatchedWith'})
            )

        return result
    
    def match_count_matrix(self, strategy=MCS.GOOD) -> np.ndarray:
        """
        Create a matrix indicating the total good matches between images.
        Reference image id is row id.
        Target image id is column id.
        Returns:
            np.ndarray:
        """
        n_images = self.n_images
        matrix = np.zeros((n_images, n_images), dtype=np.uint)

        df = self.get_number_of_matches_per_image_pair(strategy)

        matrix[
            df["ImgId"].to_numpy(),
            df["MatchedWith"].to_numpy()
        ] = df["Count"].to_numpy()

        return matrix
        
    def match_ratio_matrix(self, strategy=MCS.GOOD) -> np.ndarray:
        """
        Create a matrix indicating the ratio between total good matches between images.
        Reference image id is row id.
        Target image id is column id.
        Returns:
            np.ndarray:
        """
        n_images = self.n_images
        matrix = self.match_count_matrix().astype(np.float64)
        for id in range(n_images):
            n_ref_features = self.get_number_of_features_in_img(id)
            matrix[id] = matrix[id] / n_ref_features
        return matrix
    
    def rank_image_connections(self, strategy=MCS.GOOD) -> dict[int, list[int]]:
        temp_df = self.get_number_of_matches_per_image_pair(strategy)
        img_connections_ranked = (
            temp_df.sort_values(['ImgId', 'Count'], 
                                ascending=[True, False]).groupby('ImgId')['MatchedWith'].apply(list).to_dict()
            )
        return img_connections_ranked
    
    def get_matching_stats_string(self) -> str:
        """
        Generates the the number of matches (counts and ratio) between images
        """
        rankings = self.rank_image_connections()
        M_count = self.match_count_matrix()
        M_ratio = self.match_ratio_matrix()
        lines = []

        for id, arr in rankings.items():
            name = self.image_names[id]
            lines.append(f"{id} | {name}")
            for match_id in arr:
                count = M_count[id][match_id]
                ratio = M_ratio[id][match_id]
                lines.append(
                    f"\t{match_id} | {self.image_names[match_id]} | {count} | {ratio:.2f}"
                )
            lines.append("")

        return "\n".join(lines)