import numpy as np


class netVLAD:
    def __init__(self, kmeans, alpha=5000.0, n_cluster=32):
        self.alpha = alpha
        self.n_clusters = n_cluster
        self.kmeans = kmeans

    def Fc(self, F):
        """Computing difference Fc_ijkd[k]
        input shape: F[i,j,d] 
        output shape: Fc[i,j,k,d]
        """

        F_ijkd = np.repeat(F[:, :, np.newaxis, :], self.n_clusters, axis=2)

        c_ijkd = np.repeat(
            np.repeat(
                self.kmeans.cluster_centers_[np.newaxis, :, :],
                F.shape[1],
                axis=0,
            ),
            F.shape[0],
            axis=0,
        )

        return F_ijkd - c_ijkd

    def similarity(self, Fc_ijkd):
        """Computing similarity S_ij[k]
        input shape: F[i,j,d] 
        output shape: S[k,i,j]
        """

        norm_Fc_ijk = np.linalg.norm(Fc_ijkd, axis=3)
        exp_Fc_ijk = np.exp(-self.alpha * norm_Fc_ijk)

        S_ijk = np.divide(
            exp_Fc_ijk,
            np.repeat(
                np.sum(exp_Fc_ijk, axis=2)[:, :, np.newaxis],
                self.n_clusters,
                axis=2,
            ),
        )

        return S_ijk

    def V(self, F):
        """Computing V[d,k]
        input shape: F[i,j,d]
        output shape: V[d,k]
        """

        Fc_ijkd = self.Fc(F)
        S_ijk = self.similarity(Fc_ijkd)

        S_ijkd = np.repeat(S_ijk[:, :, :, np.newaxis], F.shape[2], axis=3)
        return np.sum(np.sum(np.multiply(S_ijkd, Fc_ijkd), axis=1), axis=0)

    def Vh(self, F):
        V_dk = self.V(F)
        Vh = V_dk.reshape((V_dk.shape[0] * V_dk.shape[1], 1))
        return Vh / np.linalg.norm(Vh)
