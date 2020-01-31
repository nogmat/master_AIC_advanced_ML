from keras.applications import VGG16
from keras.models import Sequential
from keras import layers
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
            

class discriminators:
    def __init__(self,M_l,N_l,source,target,pred, centroids, F, S):
        """
        source : array of source features extracted with G 
        target : array of target features extracted with G
        pred : array of predictions from the source array
        centroids :  the array of centroids
        F : convolutionnal features
        S : similarity to centroids
        N_l : feature map size
        M_l : feature map size
        D_h : holistic discriminor
        D_l : local discriminor
        """
        self.M_l=M_l
        self.N_l=N_l
        self.source=source
        self.target=target
        self.pred=pred
        self.centroids=centroids
        
        
        self.D_h=Sequential()
        self.D_h.add(layers.Dense(768, activation='relu', input_shape=(M_l*N_l,)))
        self.D_h.add(layers.Dense(1536, activation='relu'))
        self.D_h.add(layers.Dense(1, activation='sigmoid'))
        
        self.D_l=Sequential()
        self.D_l.add(layers.Dense(2048, activation='relu', input_shape=(M_l*N_l,)))
        self.D_l.add(layers.Dense(4096, activation='relu'))
        self.D_l.add(layers.Dense(1, activation='sigmoid'))
        
    def class_loss(self):
        ret=0
        for i in range(len(self.source)):
            ret+=self.source[i]*np.log(self.pred[i])
        return -(1/len(self.source))*ret
    
    def holistic_disc_loss(self):
        
        ret1=0
        for i in range(len(self.source)):
            ret1+=np.log(self.D_h.predict(np.array([self.source[i],])))
            
        ret2=0
        for i in range(len(self.target)):
            ret2+=np.log(1-self.D_h.predict(np.array([self.target[i],])))
         
        return -(1/len(self.source))*ret1 -(1/len(self.target))*ret2
    
    def local_disc_loss(self):
        
        ret1=0
        for n in range(len(self.source)):
            ret1_=0
            for i in range(self.M_l):
                for j in range(self.N_l):
                    ret1_+=np.log(self.D_l.predict(np.array([self.F[i,j]-self.centroids[np.argmax(self.S[:,i,j])],])))
            ret1+=(1/(self.N_l*self.M_l))*ret1_
        
        ret2=0
        for n in range(len(self.target)):
            ret2_=0
            for i in range(self.M_l):
                for j in range(self.N_l):
                    ret2_+=np.log(1-self.D_l.predict(np.array([self.F[i,j]-self.centroids[np.argmax(self.S[:,i,j])],])))
            ret2+=(1/(self.N_l*self.M_l))*ret2_
        
        return -(1/len(self.source))*ret1 -(1/len(self.target))*ret2
    
    def holistic_g_loss(self):
        ret1=0
        for i in range(len(self.target)):
            ret1+=np.log(self.D_h.predict(np.array([self.source[i],])))
        
        return -(1/len(self.target))*ret1
    
    def local_g_loss(self):
        ret1=0
        for n in range(len(self.source)):
            ret1_=0
            for i in range(self.M_l):
                for j in range(self.N_l):
                    ret1_+=np.log(self.D_l.predict(np.array([self.F[i,j]-self.centroids[np.argmax(self.S[:,i,j])],])))
            ret1+=(1/(self.N_l*self.M_l))*ret1_
        return -(1/len(self.target))*ret1

