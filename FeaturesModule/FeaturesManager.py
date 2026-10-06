from torchreid.utils import FeatureExtractor
import numpy as np
from torch.cuda import is_available


class FeaturesManager:
    def __init__(self, path_to_osnet_weights):
        self.extractor = FeatureExtractor(
            model_name='osnet_ain_x1_0',
            model_path=path_to_osnet_weights,
            device='cuda' if is_available() else 'cpu'
        )

    def extract_features(self, image_list):
        tensor_features = self.extractor(image_list)
        features = [feature.cpu().numpy() for feature in tensor_features]
        return features

    @staticmethod
    def calculate_similarity(feature_vector_1, feature_vector_2):
        eps = 1e-12
        norm_fv_1 = feature_vector_1 / (np.linalg.norm(feature_vector_1) + eps)
        norm_fv_2 = feature_vector_2 / (np.linalg.norm(feature_vector_2) + eps)

        return float(np.dot(norm_fv_1, norm_fv_2))
