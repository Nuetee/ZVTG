import numpy as np
class TCKMeans:
    def __init__(self, n_clusters=3, max_iters=100, random_state=None, temporal_window=3, alpha=0.0):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.random_state = random_state
        self.temporal_window = temporal_window // 2
        self.alpha = alpha
        self.centroids = None

    def fit_predict(self, X):
        np.random.seed(self.random_state)
        n_samples = len(X)

        # Initialize centroids randomly
        idx = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[idx]

        distances = self._calculate_distances(X)
        labels = np.argmin(distances, axis=0)
        new_labels = np.zeros_like(labels)

        for _ in range(self.max_iters):
            distances = self._calculate_distances(X)

            for i in range(n_samples):
                total_costs = []
                for k in range(self.n_clusters):
                    # Basic spatial distance to centroid
                    # spatial_cost = distances[k, i]

                    #### jinsuby ####
                    start = max(0, i - self.temporal_window)
                    end = min(n_samples, i + self.temporal_window + 1)
                    spatial_cost = np.mean(distances[k, start:end])  # avg distance over window
                    #### jinsuby ####

                    # Temporal penalty
                    temporal_cost = self._temporal_penalty(labels, i, k)
                    total_cost = spatial_cost + self.alpha * temporal_cost
                    total_costs.append(total_cost)

                new_labels[i] = np.argmin(total_costs)

            # Update centroids
            new_centroids = np.array([
                X[new_labels == k].mean(axis=0) if np.any(new_labels == k) else self.centroids[k]
                for k in range(self.n_clusters)
            ])

            if np.all(self.centroids == new_centroids):
                break

            self.centroids = new_centroids
            labels = new_labels

        return labels

    def predict(self, X):
        distances = self._calculate_distances(X)
        return np.argmin(distances, axis=0)

    def _calculate_distances(self, X):
        n_samples = X.shape[0]
        distances = np.zeros((self.n_clusters, n_samples))

        for i in range(n_samples):
            diff = self.centroids - X[i]
            distances[:, i] = np.linalg.norm(diff, axis=1)

        return distances

    def _temporal_penalty(self, labels, idx, target_cluster):
        penalty = 0
        T = len(labels)

        for offset in range(-self.temporal_window, self.temporal_window + 1):
            j = idx + offset
            if j < 0 or j >= T or j == idx:
                continue
            if labels[j] != target_cluster:
                penalty += 1

        return penalty
