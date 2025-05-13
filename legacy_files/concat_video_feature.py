import numpy as np
import os
from tqdm import tqdm

def load_and_average_features(feature_dir, save_path):
    """
    Load video features stored as .npy files, average them over the first dimension,
    and stack them together.
    
    Args:
        feature_dir (str): Directory containing .npy files, each representing a video's features.

    Returns:
        np.ndarray: Stacked feature array of shape (N * # of Frames, feature dimension).
    """
    all_features = []
    file_list = sorted(os.listdir(feature_dir))
    
    for file_name in tqdm(file_list, desc="Processing files"):
        if file_name.endswith(".npy"):
            file_path = os.path.join(feature_dir, file_name)
            features = np.load(file_path)  # Shape: (# of Frames, # of learned queries, feature dimension)
            
            # Average over the second dimension
            averaged_features = np.mean(features, axis=1)  # Shape: (# of Frames, feature dimension)
            
            all_features.append(averaged_features)
    
    # Stack all features from different videos
    final_features = np.vstack(all_features)  # Shape: (N * # of Frames, feature dimension)

    # Save the final features as a .npy file
    np.save(save_path, final_features)
    
    return final_features

# Example usage
feature_directory = "datasets/ActivityNet"
save_file = "datasets/averaged_activitynet_features.npy"
averaged_features = load_and_average_features(feature_directory, save_file)