from keras.applications import VGG16
from keras.models import Sequential
import numpy as np
from sklearn.cluster import KMeans

class Feature_Extractor:

    def __init__(self, image_size):

        self.model = Sequential()

        vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
        # vgg16.layers[-2] is block5_conv3
        # vgg16.layers[-1] is block5_pool
        for layer in vgg16.layers[:-2]:
            layer.trainable = False
        self.model.add(vgg16)

# Local Feature Pattern Learning
class LFPL:

    def __init__(self, alpha=5000.0, n_cluster=32):

        self.alpha = alpha
        self.n_clusters = n_cluster
        self.kmeans = KMeans(n_clusters = self.n_clusters)

    def similarity(self, F):
        """Computing similarity S_ij[k]
        input shape: F[i,j,d] 
        output shape: S[k,i,j]
        """
        norm_Fc_ijk = np.linalg.norm(
            np.repeat(
                F[:, :, :, np.newaxis], self.n_clusters, axis=3
            ) - self.kmeans.cluster_centers_,
            axis = 2
        )
        return np.moveaxis(
            np.exp(- self.alpha * norm_Fc_ijk),
            [0,1,2],
            [2,0,1]
        )

    def V(self, F):
        """Computing V[d,k]
        input shape: F[i,j,d] 
        output shape: V[d,k]
        """
        S_dkij = np.moveaxis(
            np.repeat(self.similarity(F)[:, :, :, np.newaxis], F.shape[2], axis=3),
            [0,1,2,3],
            [3,0,1,2]
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
