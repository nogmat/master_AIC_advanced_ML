import numpy as np


class netVLAD:

    def __init__(self, kmeans, alpha=5000.0, n_cluster=32):
        self.alpha = alpha
        self.n_clusters = n_cluster
        self.kmeans = kmeans

    def similarity(self, F):
        """Computing similarity S_ij[k]
        input shape: F[i,j,d]
        output shape: S[k,i,j]
        """
        norm_Fc_ijk = np.linalg.norm(
            np.repeat(
                F[:, :, np.newaxis, :], self.n_clusters, axis=2
            ) - np.repeat(
                np.repeat(
                    self.kmeans.cluster_centers_[np.newaxis, :, :],
                    F.shape[1], axis=0
                ),
                F.shape[0], axis=0
            ),
            axis=3
        )
        exp_Fc_ijk = np.exp(- self.alpha * norm_Fc_ijk)
        S_ijk = np.divide(
            exp_Fc_ijk,
            np.repeat(
                np.sum(exp_Fc_ijk, axis=2)[:, :, np.newaxis],
                self.n_clusters, axis=2
            )
        )
        return np.moveaxis(
            S_ijk,
            [0, 1, 2],
            [2, 0, 1]
        )

    def V(self, F):
        """Computing V[d,k]
        input shape: F[i,j,d]
        output shape: V[d,k]
        """
        S_dkij = np.moveaxis(
            np.repeat(self.similarity(
                F)[:, :, :, np.newaxis], F.shape[2], axis=3),
            [0, 1, 2, 3],
            [3, 0, 1, 2]
        )
        Fc_dkij = np.repeat(
            F[:, :, :, np.newaxis], self.n_clusters, axis=3
        ) - self.kmeans.cluster_centers_
        return np.sum(
            np.sum(
                np.multiply(S_dkij, Fc_dkij),
                axis=3
            ),
            axis=2
        )

    def Vh(self, F):
        V = self.V(F)
        Vh = V.reshape((V.shape[0]*V.shape[1]), 1))
        Vh /= np.linalg.norm(Vh)
        return V
