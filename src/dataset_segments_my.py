"""
This script defines dataset loading for the segmentation task on ABC dataset.
"""

import gc
import h5py
import numpy as np

from src.augment_utils import rotate_perturbation_point_cloud, jitter_point_cloud, shift_point_cloud, \
    random_scale_point_cloud, rotate_point_cloud, MyAugment

EPS = np.finfo(np.float32).eps


class Dataset:
    def __init__(self,
                 batch_size,
                 train_size=None,
                 val_size=None,
                 test_size=None,
                 normals=False,
                 primitives=False,
                 edges=False,
                 if_train_data=True,
                 mixnoNormal=False,
                 prefix=""):
        """
        Dataset of point cloud from ABC dataset.
        :param root_path:
        :param batch_size:
        :param if_train_data: since training dataset is large and consumes RAM,
        we can optionally choose to not load it.
        """
        self.batch_size = batch_size
        self.normals = normals
        self.primitives = primitives
        self.edges = edges
        '''
        self.augment_routines = [rotate_perturbation_point_cloud, jitter_point_cloud, shift_point_cloud,
                                 random_scale_point_cloud, rotate_point_cloud]
        '''
        self.augment_routines = MyAugment()
        self.mixnoNormal = mixnoNormal

        if if_train_data:
            with h5py.File("/data/train_data_withEdge.h5", "r") as hf:
                train_points = np.array(hf.get("points"))
                train_labels = np.array(hf.get("labels"))
                if normals:
                    train_normals = np.array(hf.get("normals"))
                if primitives:
                    train_primitives = np.array(hf.get("prim"))
                if edges:
                    with h5py.File("/data/train_My_Edge.h5", "r") as hf_edge:
                        self.train_edges = np.array(hf_edge.get("label")).astype(np.long)  # [B, N] 
                        self.train_edges_W = np.array(hf_edge.get("W")).astype(np.float32)
            train_points = train_points[0:train_size].astype(np.float32)
            self.train_labels = train_labels[0:train_size]
            if normals:
                self.train_normals = train_normals[0:train_size].astype(np.float32)
            if primitives:
                self.train_primitives = train_primitives[0:train_size]
            if edges:
                self.train_edges = self.train_edges[0:train_size]
                self.train_edges_W = self.train_edges_W[0:train_size]
            means = np.mean(train_points, 1)
            means = np.expand_dims(means, 1)

            self.train_points = (train_points - means)
        
        print("load train data")

        with h5py.File("/data/test_data_withEdge.h5", "r") as hf:
            test_points = np.array(hf.get("points"))
            test_labels = np.array(hf.get("labels"))
            if normals:
                test_normals = np.array(hf.get("normals"))
            if primitives:
                test_primitives = np.array(hf.get("prim"))
            if edges:
                with h5py.File("/data/test_My_Edge.h5", "r") as hf_edge:
                    self.test_edges = np.array(hf_edge.get("label")).astype(np.long)  # [B, N] 
                    self.test_edges_W = np.array(hf_edge.get("W")).astype(np.float32)

        test_points = test_points[0:test_size].astype(np.float32)
        test_labels = test_labels[0:test_size]

        if normals:
            # self.val_normals = val_normals[0:val_size].astype(np.float32)
            self.test_normals = test_normals[0:test_size].astype(np.float32)

        if primitives:
            # self.val_primitives = val_primitives[0:val_size]
            self.test_primitives = test_primitives[0:test_size]
        if edges:
            self.test_edges = self.test_edges[0:test_size]
            self.test_edges_W = self.test_edges_W[0:test_size]

        means = np.mean(test_points, 1)
        means = np.expand_dims(means, 1)
        self.test_points = (test_points - means)
        self.test_labels = test_labels

        print("load test data")
        gc.collect()

    def get_train(self, randomize=False, augment=False, anisotropic=False, align_canonical=False,
                  if_normal_noise=False, if_jitter_points=False):
        train_size = self.train_points.shape[0]
        while (True):
            l = np.arange(train_size)
            if randomize:
                np.random.shuffle(l)
            train_points = self.train_points[l]
            train_labels = self.train_labels[l]

            if self.normals:
                train_normals = self.train_normals[l]
            if self.primitives:
                train_primitives = self.train_primitives[l]
            if self.edges:
                train_edges = self.train_edges[l]
                train_edges_W = self.train_edges_W[l]

            for i in range(train_size // self.batch_size):
                points = train_points[i * self.batch_size:(i + 1) *
                                                          self.batch_size]
                if self.normals:
                    normals = train_normals[i * self.batch_size:(i + 1) * self.batch_size]

                if augment:
                    # points = self.augment_routines[np.random.choice(np.arange(5))](points)
                    if not self.normals:
                        points = self.augment_routines.augment(points)
                    else:
                        points,normals = self.augment_routines.augment([points, normals])

                if if_normal_noise:
                    normals = train_normals[i * self.batch_size:(i + 1) * self.batch_size]
                    noise = normals * np.clip(np.random.randn(1, points.shape[1], 1) * 0.01, a_min=-0.01, a_max=0.01)
                    points = points + noise.astype(np.float32)
                
                # ==========================================================
                if self.mixnoNormal:
                    random_choice = np.random.choice(np.arange(self.batch_size), self.batch_size // 2, replace=False)
                    normals[random_choice, :, :] = 0

                labels = train_labels[i * self.batch_size:(i + 1) * self.batch_size]

                for j in range(self.batch_size):
                    if align_canonical:
                        S, U = self.pca_numpy(points[j])
                        smallest_ev = U[:, np.argmin(S)]
                        R = self.rotation_matrix_a_to_b(smallest_ev, np.array([1, 0, 0]))
                        # rotate input points such that the minor principal
                        # axis aligns with x axis.
                        points[j] = (R @ points[j].T).T

                        if self.normals:
                            normals[j] = (R @ normals[j].T).T

                        std = np.max(points[j], 0) - np.min(points[j], 0)
                        if anisotropic:
                            points[j] = points[j] / (std.reshape((1, 3)) + EPS)
                            # TODO make the same changes to normals also.
                        else:
                            points[j] = points[j] / (np.max(std) + EPS)

                if if_jitter_points:
                    noise = np.random.uniform(low=-0.5, high=0.5, size=(1, points.shape[1], 3)).astype(np.float32) * 0.024
                    #print(noise.max(), noise.min(), points.max(), points.min())
                    points = points + noise
                    normals = normals + noise

                return_items = [points, labels]
                if self.normals:
                    return_items.append(normals)
                else:
                    return_items.append(None)

                if self.primitives:
                    primitives = train_primitives[i * self.batch_size:(i + 1) * self.batch_size]
                    return_items.append(primitives)
                else:
                    return_items.append(None)

                if self.edges:
                    edges = train_edges[i * self.batch_size:(i + 1) * self.batch_size]
                    return_items.append(edges)
                    edges_W = train_edges_W[i * self.batch_size:(i + 1) * self.batch_size]
                    return_items.append(edges_W)
                else:
                    return_items.append(None)
                    return_items.append(None)

                yield return_items

    def get_test(self, randomize=False, anisotropic=False, align_canonical=False, if_normal_noise=False, normal_zero=False):
        test_size = self.test_points.shape[0]
        batch_size = self.batch_size

        while (True):
            for i in range(test_size // batch_size):
                points = self.test_points[i * self.batch_size:(i + 1) *
                                                              self.batch_size]
                labels = self.test_labels[i * self.batch_size:(i + 1) * self.batch_size]
                if self.normals:
                    normals = self.test_normals[i * self.batch_size:(i + 1) *
                                                                    self.batch_size]
                if if_normal_noise and self.normals:
                    normals = self.test_normals[i * self.batch_size:(i + 1) *
                                                                    self.batch_size]
                    noise = normals * np.clip(np.random.randn(1, points.shape[1], 1) * 0.01, a_min=-0.01, a_max=0.01)
                    points = points + noise.astype(np.float32)

                new_points = []
                for j in range(self.batch_size):
                    if align_canonical:
                        S, U = self.pca_numpy(points[j])
                        smallest_ev = U[:, np.argmin(S)]
                        R = self.rotation_matrix_a_to_b(smallest_ev, np.array([1, 0, 0]))
                        # rotate input points such that the minor principal
                        # axis aligns with x axis.
                        points[j] = (R @ points[j].T).T
                        if self.normals:
                            normals[j] = (R @ normals[j].T).T

                        std = np.max(points[j], 0) - np.min(points[j], 0)
                        if anisotropic:
                            points[j] = points[j] / (std.reshape((1, 3)) + EPS)
                        else:
                            points[j] = points[j] / (np.max(std) + EPS)

                return_items = [points, labels]
                if self.normals:
                    if normal_zero:
                        normals = normals * 0
                    return_items.append(normals)
                else:
                    return_items.append(None)

                if self.primitives:
                    primitives = self.test_primitives[i * self.batch_size:(i + 1) * self.batch_size]
                    return_items.append(primitives)
                else:
                    return_items.append(None)

                if self.edges:
                    edges = self.test_edges[i * self.batch_size:(i + 1) * self.batch_size]
                    return_items.append(edges)
                    edges_W = self.test_edges_W[i * self.batch_size:(i + 1) * self.batch_size]
                    return_items.append(edges_W)
                else:
                    return_items.append(None)
                    return_items.append(None)   

                yield return_items


    def get_val(self, randomize=False, anisotropic=False, align_canonical=False, if_normal_noise=False):
        val_size = self.val_points.shape[0]
        batch_size = self.batch_size

        while (True):
            for i in range(val_size // batch_size):
                points = self.val_points[i * self.batch_size:(i + 1) *
                                                             self.batch_size]
                labels = self.val_labels[i * self.batch_size:(i + 1) * self.batch_size]
                if self.normals:
                    normals = self.val_normals[i * self.batch_size:(i + 1) *
                                                                   self.batch_size]
                if if_normal_noise and self.normals:
                    normals = self.val_normals[i * self.batch_size:(i + 1) *
                                                                   self.batch_size]
                    noise = normals * np.clip(np.random.randn(1, points.shape[1], 1) * 0.01, a_min=-0.01, a_max=0.01)
                    points = points + noise.astype(np.float32)

                new_points = []
                for j in range(self.batch_size):
                    if align_canonical:
                        S, U = self.pca_numpy(points[j])
                        smallest_ev = U[:, np.argmin(S)]
                        R = self.rotation_matrix_a_to_b(smallest_ev, np.array([1, 0, 0]))
                        # rotate input points such that the minor principal
                        # axis aligns with x axis.
                        points[j] = (R @ points[j].T).T
                        if self.normals:
                            normals[j] = (R @ normals[j].T).T

                        std = np.max(points[j], 0) - np.min(points[j], 0)
                        if anisotropic:
                            points[j] = points[j] / (std.reshape((1, 3)) + EPS)
                        else:
                            points[j] = points[j] / (np.max(std) + EPS)

                return_items = [points, labels]
                if self.normals:
                    return_items.append(normals)
                else:
                    return_items.append(None)

                if self.primitives:
                    primitives = self.val_primitives[i * self.batch_size:(i + 1) * self.batch_size]
                    return_items.append(primitives)
                else:
                    return_items.append(None)
                yield return_items

    def normalize_points(self, points, normals, anisotropic=False):
        points = points - np.mean(points, 0, keepdims=True)
        noise = normals * np.clip(np.random.randn(points.shape[0], 1) * 0.01, a_min=-0.01, a_max=0.01)
        points = points + noise.astype(np.float32)

        S, U = self.pca_numpy(points)
        smallest_ev = U[:, np.argmin(S)]
        R = self.rotation_matrix_a_to_b(smallest_ev, np.array([1, 0, 0]))
        # rotate input points such that the minor principal
        # axis aligns with x axis.
        points = (R @ points.T).T
        normals = (R @ normals.T).T
        std = np.max(points, 0) - np.min(points, 0)
        if anisotropic:
            points = points / (std.reshape((1, 3)) + EPS)
        else:
            points = points / (np.max(std) + EPS)
        return points.astype(np.float32), normals.astype(np.float32)

    def rotation_matrix_a_to_b(self, A, B):
        """
        Finds rotation matrix from vector A in 3d to vector B
        in 3d.
        B = R @ A
        """
        cos = np.dot(A, B)
        sin = np.linalg.norm(np.cross(B, A))
        u = A
        v = B - np.dot(A, B) * A
        v = v / (np.linalg.norm(v) + EPS)
        w = np.cross(B, A)
        w = w / (np.linalg.norm(w) + EPS)
        F = np.stack([u, v, w], 1)
        G = np.array([[cos, -sin, 0],
                      [sin, cos, 0],
                      [0, 0, 1]])
        # B = R @ A
        try:
            R = F @ G @ np.linalg.inv(F)
        except:
            R = np.eye(3, dtype=np.float32)
        return R

    def pca_numpy(self, X):
        S, U = np.linalg.eig(X.T @ X)
        return S, U



from torch.utils.data import Dataset as Dataset_std

class my_simple_data(Dataset_std):
    def __init__(self, ids=[], prefix="", if_normals=False, if_train=False, ret_edges1w=False, aug=True, noise=False, noise_level=0):
        super().__init__()

        self.ret_edges1w = ret_edges1w
        self.ids=ids
        self.if_normals = if_normals
        
        total_ids = np.loadtxt(prefix + "data/{}_new_ids.txt".format("train" if if_train else "test"), delimiter=" ", dtype=np.int32)[:, 1]  # [N, ]
        # print(total_ids)
        self.rows = np.arange(total_ids.shape[0])

        self.length = self.rows.shape[0]
        self.if_train = if_train
        self.prefix = prefix
        self.myAug = MyAugment()
        self.aug = aug
        if self.aug:
            print("my dataset use data aug")
        
        self.noise = noise
        if self.noise:
            print("my dataset use noise with level", noise_level)
        self.noise_level=noise_level        

    def load_hdf5(self, ):
        with h5py.File(self.prefix + "data/{}_data_withEdge.h5".format("train" if self.if_train else "test"), "r") as hf:
            # print("hdf5 load ok")
            points = np.array(hf.get("points"))
            labels = np.array(hf.get("labels"))
            if self.if_normals:
                normals = np.array(hf.get("normals"))
                self.normals = normals[self.rows].astype(np.float32)
                
            if self.ret_edges1w:
                edges1w = np.array(hf.get("edge"))  # ==================== 
                self.edges1w = edges1w[self.rows].astype(np.float32)
                print(self.edges1w.shape)

            primitives = np.array(hf.get("prim"))
        points = points[self.rows].astype(np.float32)
        labels = labels[self.rows]
        
        self.primitives = primitives[self.rows]
        means = np.mean(points, 1)
        means = np.expand_dims(means, 1)

        self.points = (points - means)
        
        if self.ret_edges1w:   
            self.edges1w = self.edges1w - means
        self.labels = labels

        with h5py.File(self.prefix + "data/train_My_Edge.h5", "r") as hf_edge:
            # print("edge load ok")
            self.train_edges = np.array(hf_edge.get("label")).astype(np.compat.long)  # [B, N] 
            self.train_edges_W = np.array(hf_edge.get("W")).astype(np.float32)

    def __len__(self,):
        return self.length

    def __getitem__(self, index):
        if not hasattr(self, 'points'):
            self.load_hdf5()
        _points = self.points[index]
        _labels = self.labels[index]
        _prims = self.primitives[index]
        
        if self.if_normals:
            _normals = self.normals[index]
        
        if self.ret_edges1w:
            _edges1w = self.edges1w[index]
        
        std = np.max(_points, 0) - np.min(_points, 0)

        _points = _points / (np.max(std) + EPS)

        if self.ret_edges1w:
            _edges1w = _edges1w / (np.max(std) + EPS)


        if self.if_train and self.aug:
            if not self.if_normals:
                _points = self.myAug.augment(_points.reshape(1, 10000, 3))[0]
            else:
                tmp = [_points.reshape(1, 10000, 3), ]
                if self.ret_edges1w:
                    tmp.append(_edges1w.reshape(1, 10000, 3))
                tmp.append(_normals.reshape(1, 10000, 3))
                tmp = self.myAug.augment(tmp)  # ([_points.reshape(1, 10000, 3), _normals.reshape(1, 10000, 3)])
                _points = tmp[0][0]
                if self.ret_edges1w:
                    _edges1w = tmp[1][0]
                _normals = tmp[-1][0]
        
        S, U = self.pca_numpy(_points)
        smallest_ev = U[:, np.argmin(S)]
        R = self.rotation_matrix_a_to_b(smallest_ev, np.array([1, 0, 0]))
        _points = (R @ _points.T).T
        if self.if_normals:
            _normals = (R @ _normals.T).T
        if self.ret_edges1w:
            _edges1w = (R @ _edges1w.T).T
        
        if self.noise and self.noise_level != -1:
            flag_noise = self.noise_level
            if flag_noise == 0:
                sigma = 0.005
            elif flag_noise == 1:
                sigma=0.01
            elif flag_noise == 2:
                sigma =0.02
                # print('noise level 2')
            elif flag_noise == 3:
                sigma = 0.05 

            clip= 5.0 * sigma
            jittered_data_pts = np.clip(sigma * np.random.randn(_points.shape[0],3), -1 * clip, clip)
            _points[:,:3] = _points[:,:3] + jittered_data_pts
        
        elif self.noise and self.noise_level == -1:
            
            w = np.random.random((_normals.shape[0], 1)) 
            shift = np.clip(0.087 * np.random.randn(_normals.shape[0], 1), -3 * 0.087, 0.087 * 3) 
            angle2 = np.arctan(_normals[:, 0] / (_normals[:, 1] + 1e-8))
            a1 = np.zeros(_normals.shape)
            a1[:, 0], a1[:, 1] = np.cos(angle2), np.sin(angle2)
            a2 = np.cross(a1, _normals)
            _normals = _normals + (w * a1 + (1-w) * a2) * shift

            sigma = 0.025
            _points = np.clip(sigma * 0.33 * np.random.randn(_points.shape[0],1), -sigma , sigma ) * _normals + _points
        return_items = [_points.astype(np.float32), _labels]

        if self.if_normals:
                return_items.append(_normals.astype(np.float32))
        else:
                return_items.append(np.zeros((1,)).astype(np.float32))
        
        return_items.append(_prims.astype(np.int64))

        return_items.append(self.train_edges[index])
        return_items.append(self.train_edges_W[index].astype(np.float32))

        if self.ret_edges1w:
            return_items.append(_edges1w.astype(np.float32))
                
        if self.if_train:
            return self.random_points(return_items)
        return return_items

    def random_points(self, items):
            l = np.arange(10000)
            np.random.shuffle(l)
            ret = []
            for item in items:
                if item.shape[0] == 10000:
                    ret.append(item[l])
                else:
                    ret.append(item)
            return ret

    def pca_numpy(self, X):
        S, U = np.linalg.eig(X.T @ X)
        return S, U

    def rotation_matrix_a_to_b(self, A, B):
        """
        Finds rotation matrix from vector A in 3d to vector B
        in 3d.
        B = R @ A
        """
        cos = np.dot(A, B)
        sin = np.linalg.norm(np.cross(B, A))
        u = A
        v = B - np.dot(A, B) * A
        v = v / (np.linalg.norm(v) + EPS)
        w = np.cross(B, A)
        w = w / (np.linalg.norm(w) + EPS)
        F = np.stack([u, v, w], 1)
        G = np.array([[cos, -sin, 0],
                      [sin, cos, 0],
                      [0, 0, 1]])
        # B = R @ A
        try:
            R = F @ G @ np.linalg.inv(F)
        except:
            R = np.eye(3, dtype=np.float32)
        return R    


if __name__ == "__main__":
    mix_train_dataset = my_simple_data(prefix="/data/ytliu/parsenet/", if_normals=True, if_train=False, ret_edges1w=False, aug=False, noise=True, noise_level=-1)
    for i in range(20):
        points, normals = mix_train_dataset.__getitem__(i)[0],mix_train_dataset.__getitem__(i)[2]
        print(points.shape)
        np.savetxt(f"./debug_noisev2/{i}.txt", np.concatenate((points, normals), axis=-1), fmt="%0.4f", delimiter=";")
