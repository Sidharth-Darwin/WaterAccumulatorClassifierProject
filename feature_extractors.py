import numpy as np

class FeatureExtractor:
    def extract_features(self, image, top_pour_left, top_pour_right, bottom_pour_left, bottom_pour_right, left_pour_top, left_pour_bottom, right_pour_top, right_pour_bottom, full_pour):
        all_features = []
        full_pour_out = full_pour.output
        all_features.append(np.bitwise_xor(top_pour_left.output, full_pour_out[:, :full_pour_out.shape[1]//2]))
        all_features.append(np.bitwise_xor(top_pour_right.output, full_pour_out[:, full_pour_out.shape[1]//2:]))
        all_features.append(np.bitwise_xor(bottom_pour_left.output, full_pour_out[:, :full_pour_out.shape[1]//2]))
        all_features.append(np.bitwise_xor(bottom_pour_right.output, full_pour_out[:, full_pour_out.shape[1]//2:]))
        all_features.append(np.bitwise_xor(left_pour_top.output, full_pour_out[:full_pour_out.shape[0]//2, :]))
        all_features.append(np.bitwise_xor(left_pour_bottom.output, full_pour_out[full_pour_out.shape[0]//2:, :]))
        all_features.append(np.bitwise_xor(right_pour_top.output, full_pour_out[:full_pour_out.shape[0]//2, :]))
        all_features.append(np.bitwise_xor(right_pour_bottom.output, full_pour_out[full_pour_out.shape[0]//2:, :]))
        all_features.append(full_pour_out)

        all_features = [np.count_nonzero(feature) for feature in all_features]
        all_features = [feature / image.size for feature in all_features] + [np.count_nonzero(image) / image.size]

        return np.array(all_features, dtype=np.float32)