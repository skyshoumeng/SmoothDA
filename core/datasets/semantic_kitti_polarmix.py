import os
import os.path

import numpy as np
from torchsparse import SparseTensor
from torchsparse.utils.collate import sparse_collate_fn
from torchsparse.utils.quantize import sparse_quantize
from torchpack.utils.logging import logger

from core.datasets.utils import polarmix
import glob
import pickle
import copy
import cv2
__all__ = ['SemanticKITTI_PolarMix']

label_name_mapping = {
    0: 'unlabeled',
    1: 'outlier',
    10: 'car',
    11: 'bicycle',
    13: 'bus',
    15: 'motorcycle',
    16: 'on-rails',
    18: 'truck',
    20: 'other-vehicle',
    30: 'person',
    31: 'bicyclist',
    32: 'motorcyclist',
    40: 'road',
    44: 'parking',
    48: 'sidewalk',
    49: 'other-ground',
    50: 'building',
    51: 'fence',
    52: 'other-structure',
    60: 'lane-marking',
    70: 'vegetation',
    71: 'trunk',
    72: 'terrain',
    80: 'pole',
    81: 'traffic-sign',
    99: 'other-object',
    252: 'moving-car',
    253: 'moving-bicyclist',
    254: 'moving-person',
    255: 'moving-motorcyclist',
    256: 'Moving-on-rails',
    257: 'moving-bus',
    258: 'moving-truck',
    259: 'moving-other-vehicle'
}

kept_labels = [
    'road', 'sidewalk', 'parking', 'other-ground', 'building', 'car', 'truck',
    'bicycle', 'motorcycle', 'other-vehicle', 'vegetation', 'trunk', 'terrain',
    'person', 'bicyclist', 'motorcyclist', 'fence', 'pole', 'traffic-sign'
]

"""
{'car': 0, 'bicycle': 1, 'motorcycle': 2, 'truck': 3, 'other-vehicle': 4, 'person': 5, 'bicyclist': 6, 'motorcyclist': 7, 
'road': 8, 'parking': 9, 'sidewalk': 10, 'other-ground': 11, 'building': 12, 'fence': 13, 'vegetation': 14, 'trunk': 15, 
'terrain': 16, 'pole': 17, 'traffic-sign': 18}
"""
instance_classes = [0, 1, 2, 3, 4, 5, 6, 7]
Omega = [np.random.random() * np.pi * 2 / 3, (np.random.random() + 1) * np.pi * 2 / 3]  # x3


class PointWOLF(object):
    def __init__(self, num_anchor=20):
        self.num_anchor = 4
        self.sample_type = 'fps'
        self.sigma = 0.5

        w_R_range = 10
        w_S_range = 3
        w_T_range = 0.25
        self.R_range = (-abs(w_R_range), abs(w_R_range))
        self.S_range = (1., w_S_range)
        self.T_range = (-abs(w_T_range), abs(w_T_range))
        
        
    def __call__(self, pos):
        """
        input :
            pos([N,3])
            
        output : 
            pos([N,3]) : original pointcloud
            pos_new([N,3]) : Pointcloud augmneted by PointWOLF
        """
        M=self.num_anchor #(Mx3)
        N, _=pos.shape #(N)
        
        if self.sample_type == 'random':
            idx = np.random.choice(N,M)#(M)
        elif self.sample_type == 'fps':
            idx = self.fps(pos, M) #(M)
        
        pos_anchor = pos[idx] #(M,3), anchor point
        
        pos_repeat = np.expand_dims(pos,0).repeat(M, axis=0)#(M,N,3)
        pos_normalize = np.zeros_like(pos_repeat, dtype=pos.dtype)  #(M,N,3)
        
        #Move to canonical space
        pos_normalize = pos_repeat - pos_anchor.reshape(M,-1,3)
        
        #Local transformation at anchor point
        pos_transformed = self.local_transformaton(pos_normalize) #(M,N,3)
        
        #Move to origin space
        pos_transformed = pos_transformed + pos_anchor.reshape(M,-1,3) #(M,N,3)
        
        pos_new = self.kernel_regression(pos, pos_anchor, pos_transformed)
        pos_new = self.normalize(pos_new)
        
        return pos.astype('float32'), pos_new.astype('float32')
        

    def kernel_regression(self, pos, pos_anchor, pos_transformed):
        """
        input :
            pos([N,3])
            pos_anchor([M,3])
            pos_transformed([M,N,3])
            
        output : 
            pos_new([N,3]) : Pointcloud after weighted local transformation 
        """
        M, N, _ = pos_transformed.shape
        
        #Distance between anchor points & entire points
        sub = np.expand_dims(pos_anchor,1).repeat(N, axis=1) - np.expand_dims(pos,0).repeat(M, axis=0) #(M,N,3), d
        
        project_axis = self.get_random_axis(1)

        projection = np.expand_dims(project_axis, axis=1)*np.eye(3)#(1,3,3)
        
        #Project distance
        sub = sub @ projection # (M,N,3)
        sub = np.sqrt(((sub) ** 2).sum(2)) #(M,N)  
        
        #Kernel regression
        weight = np.exp(-0.5 * (sub ** 2) / (self.sigma ** 2))  #(M,N) 
        pos_new = (np.expand_dims(weight,2).repeat(3, axis=-1) * pos_transformed).sum(0) #(N,3)
        pos_new = (pos_new / weight.sum(0, keepdims=True).T) # normalize by weight
        return pos_new

    
    def fps(self, pos, npoint):
        """
        input : 
            pos([N,3])
            npoint(int)
            
        output : 
            centroids([npoints]) : index list for fps
        """
        N, _ = pos.shape
        centroids = np.zeros(npoint, dtype=np.int_) #(M)
        distance = np.ones(N, dtype=np.float64) * 1e10 #(N)
        farthest = np.random.randint(0, N, (1,), dtype=np.int_)
        for i in range(npoint):
            centroids[i] = farthest
            centroid = pos[farthest, :]
            dist = ((pos - centroid)**2).sum(-1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = distance.argmax()
        return centroids
    
    def local_transformaton(self, pos_normalize):
        """
        input :
            pos([N,3]) 
            pos_normalize([M,N,3])
            
        output :
            pos_normalize([M,N,3]) : Pointclouds after local transformation centered at M anchor points.
        """
        M,N,_ = pos_normalize.shape
        transformation_dropout = np.random.binomial(1, 0.5, (M,3)) #(M,3)
        transformation_axis =self.get_random_axis(M) #(M,3)

        degree = np.pi * np.random.uniform(*self.R_range, size=(M,3)) / 180.0 * transformation_dropout[:,0:1] #(M,3), sampling from (-R_range, R_range) 
        
        scale = np.random.uniform(*self.S_range, size=(M,3)) * transformation_dropout[:,1:2] #(M,3), sampling from (1, S_range)
        scale = scale*transformation_axis
        scale = scale + 1*(scale==0) #Scaling factor must be larger than 1
        
        trl = np.random.uniform(*self.T_range, size=(M,3)) * transformation_dropout[:,2:3] #(M,3), sampling from (1, T_range)
        trl = trl*transformation_axis
        
        #Scaling Matrix
        S = np.expand_dims(scale, axis=1)*np.eye(3) # scailing factor to diagonal matrix (M,3) -> (M,3,3)
        #Rotation Matrix
        sin = np.sin(degree)
        cos = np.cos(degree)
        sx, sy, sz = sin[:,0], sin[:,1], sin[:,2]
        cx, cy, cz = cos[:,0], cos[:,1], cos[:,2]
        R = np.stack([cz*cy, cz*sy*sx - sz*cx, cz*sy*cx + sz*sx,
             sz*cy, sz*sy*sx + cz*cy, sz*sy*cx - cz*sx,
             -sy, cy*sx, cy*cx], axis=1).reshape(M,3,3)
        
        pos_normalize = pos_normalize@R@S + trl.reshape(M,1,3)
        return pos_normalize
    
    def get_random_axis(self, n_axis):
        """
        input :
            n_axis(int)
            
        output :
            axis([n_axis,3]) : projection axis   
        """
        axis = np.random.randint(1,8, (n_axis)) # 1(001):z, 2(010):y, 3(011):yz, 4(100):x, 5(101):xz, 6(110):xy, 7(111):xyz    
        m = 3 
        axis = (((axis[:,None] & (1 << np.arange(m)))) > 0).astype(int)
        return axis
    
    def normalize(self, pos):
        """
        input :
            pos([N,3])
        
        output :
            pos([N,3]) : normalized Pointcloud
        """
        pos = pos - pos.mean(axis=-2, keepdims=True)
        scale = (1 / np.sqrt((pos ** 2).sum(1)).max()) * 0.999999
        pos = scale * pos
        return pos



class SemanticKITTI_PolarMix(dict):

    def __init__(self, root, voxel_size, num_points, **kwargs):
        submit_to_server = kwargs.get('submit', False)
        sample_stride = kwargs.get('sample_stride', 1)
        google_mode = kwargs.get('google_mode', False)

        logger.info("SemanticKITTI with PolarMix\n")

        if submit_to_server or False:
            super().__init__({
                'train':
                    SemanticKITTIInternal(root,
                                          voxel_size,
                                          num_points,
                                          sample_stride=1,
                                          split='train',
                                          submit=True),
                'test':
                    SemanticKITTIInternal(root,
                                          voxel_size,
                                          num_points,
                                          sample_stride=1,
                                          split='test')
            })
        else:
            super().__init__({
                'train':
                    SemanticKITTIInternal(root,
                                          voxel_size,
                                          num_points,
                                          polarcutmix=True,
                                          sample_stride=1,
                                          split='train',
                                          google_mode=google_mode),
                'test':
                    SemanticKITTIInternal(root,
                                          voxel_size,
                                          num_points,
                                          sample_stride=sample_stride,
                                          split='val')
            })





class SemanticKITTIInternal:

    def __init__(self,
                 root,
                 voxel_size,
                 num_points,
                 split,
                 polarcutmix=False,
                 sample_stride=1,
                 submit=False,
                 google_mode=True):
        if submit:
            trainval = True
        else:
            trainval = False
        self.root = root
        self.split = split
        self.voxel_size = voxel_size
        self.num_points = num_points
        self.polarcutmix = polarcutmix
        self.sample_stride = sample_stride
        self.google_mode = google_mode
        self.seqs = []
        if split == 'train':
            self.seqs = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']
            if self.google_mode or trainval:
                self.seqs.append('08')
        elif self.split == 'val':
            self.seqs = ['08']
        elif self.split == 'test':
            self.seqs = [
                '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21'
            ]

        self.files = []
        for seq in self.seqs:
            seq_files = sorted(
                os.listdir(os.path.join(self.root, seq, 'velodyne')))
            seq_files = [
                os.path.join(self.root, seq, 'velodyne', x) for x in seq_files
            ]
            self.files.extend(seq_files)

        if self.sample_stride > 1:
            self.files = self.files[::self.sample_stride]

        reverse_label_name_mapping = {}
        self.label_map = np.zeros(260)
        cnt = 0
        for label_id in label_name_mapping:
            if label_id > 250:
                if label_name_mapping[label_id].replace('moving-',
                                                        '') in kept_labels:
                    self.label_map[label_id] = reverse_label_name_mapping[
                        label_name_mapping[label_id].replace('moving-', '')]
                else:
                    self.label_map[label_id] = 255
            elif label_id == 0:
                self.label_map[label_id] = 255
            else:
                if label_name_mapping[label_id] in kept_labels:
                    self.label_map[label_id] = cnt
                    reverse_label_name_mapping[
                        label_name_mapping[label_id]] = cnt
                    cnt += 1
                else:
                    self.label_map[label_id] = 255

        self.reverse_label_name_mapping = reverse_label_name_mapping
        self.num_classes = cnt
        self.angle = 0.0
        self.inst_path_label = np.load("path_label_dict.pkl", allow_pickle=True)
        self.count = 0

        self.voxel = np.zeros((100, 100, 5))

        self.maps = np.zeros((112,112))


        self.pointwolf = PointWOLF()


    def set_angle(self, angle):
        self.angle = angle

    def __len__(self):
        return len(self.files)

    def read_lidar_scan(self, index):
        with open(self.files[index], 'rb') as b:
            block_ = np.fromfile(b, dtype=np.float32).reshape(-1, 4)
        label_file = self.files[index].replace('velodyne', 'labels').replace('.bin', '.label')
        if os.path.exists(label_file):
            with open(label_file, 'rb') as a:
                all_labels = np.fromfile(a, dtype=np.int32).reshape(-1)
        else:
            all_labels = np.zeros(block_.shape[0]).astype(np.int32)
        labels_ = self.label_map[all_labels & 0xFFFF].astype(np.int64)
        return block_, labels_

    def twis(self, points):
        ## setting differently for better performance
        cxyzr = np.mean(points, axis=0)
        points = points - cxyzr
        
        xyz = points[:,:3]
        if np.random.randn(1) > .5:
            delta_x = np.cos( xyz[:,1] / (0.5 + np.random.rand(1) * 1.5) + np.random.randn(1) * 2 * np.pi ) * ((np.random.rand(1)-0.5) * 1.)
            xyz[:,0] += delta_x

        if np.random.randn(1) > .5:
            delta_y = np.cos( xyz[:,0] / (0.5 + np.random.rand(1) * 1.5) + np.random.randn(1) * 2 * np.pi ) * ((np.random.rand(1)-0.5) * 1.)
            xyz[:,1] += delta_y

        if np.random.randn(1) > .5:
            rho = np.sqrt(xyz[:,0]**2 + xyz[:,1]**2)
            delta_z = np.cos( rho / (0.5 + np.random.rand(1) * 1.5) + np.random.randn(1) * 2 * np.pi ) * ((np.random.rand(1)-0.5) * .5)
            xyz[:,2] += delta_z 
        points[:,:3] = xyz
        
        return  points + cxyzr


    def farthest_point_sample(self, xyz, npoint): 
        """
        Input:
            xyz: pointcloud data, [B, N, 3]
            npoint: number of samples
        Return:
            centroids: sampled pointcloud index, [B, npoint]
        """
        xyz = np.expand_dims(xyz, 0)
        #xyz = xyz.transpose(0,2,1)
        B, N, C = xyz.shape
        
        centroids = np.zeros((B, npoint))    
        distance = np.ones((B, N)) * 1e10                 

        batch_indices = np.arange(B)       
        
        barycenter = np.sum((xyz), 1)                                  
        barycenter = barycenter/xyz.shape[1]
        barycenter = barycenter.reshape(B, 1, C) 

        dist = np.sum((xyz - barycenter) ** 2, -1)
        farthest = np.argmax(dist,1)                                 

        samples = []
        for i in range(npoint):
            #print("-------------------------------------------------------")
            #print("The %d farthest pts %s " % (i, farthest))
            centroids[:, i] = farthest                                    
            centroid = xyz[batch_indices, farthest, :].reshape(B, 1, C)       
            samples.append(centroid)
            dist = np.sum((xyz - centroid) ** 2, -1)                
            #print("dist    : ", dist)
            mask = dist < distance
            #print("mask %i : %s" % (i,mask))
            distance[mask] = dist[mask]                                  
            #print("distance: ", distance)
            farthest = np.argmax(distance, -1)                      
    
        return samples


    def thin_road(self, points, labels):
        maps = self.maps * 0.0
        tmp_points = np.clip(points, np.random.randint(-45, -30), np.random.randint(30, 45))
        tmp_p = tmp_points[labels != 8]
        maps[np.floor((tmp_p[:,0]+51)*1).astype(np.int32), np.floor((tmp_p[:,1]+51)*1).astype(np.int32)] = 1.
        maps = cv2.blur(maps, (3, 3))
        mask = maps[np.floor((tmp_points[:,0]+51)*1).astype(np.int32), np.floor((tmp_points[:,1]+51)*1).astype(np.int32)] < 1e-2

        return points[mask]


    def twis_scene(self, points):
        ## setting differently for better performance
        xyz = points[:,:3]
        if np.random.randn(1) > .5:
            delta_x = np.cos( xyz[:,1] / (5. + np.random.rand(1) * 15) + np.random.rand(1) * 2 * np.pi ) * ((np.random.rand(1)-0.5) * 10.)
            xyz[:,0] += delta_x

        if np.random.randn(1) > .5:
            delta_y = np.cos( xyz[:,0] / (5. + np.random.rand(1) * 15) + np.random.rand(1) * 2 * np.pi ) * ((np.random.rand(1)-0.5) * 10.)
            xyz[:,1] += delta_y

        if np.random.randn(1) > .5:
            rho = np.sqrt(xyz[:,0]**2 + xyz[:,1]**2)
            delta_z = np.cos( rho / (5. + np.random.rand(1) * 15) + np.random.rand(1) * 2 * np.pi ) * ((np.random.rand(1)-0.5) * 1.)
            xyz[:,2] += delta_z
        points[:,:3] = xyz

        return  points 


    def __getitem__(self, index):
        block_, labels_ = self.read_lidar_scan(index)


        #print(" ^ " * 10)
        if self.split == 'train' and (np.random.rand(1) > 0.2):
            # read another lidar scan
            #index_2 = np.random.randint(len(self.files))
            #pts2, labels2 = self.read_lidar_scan(index_2)
                     
            names = self.files[index].split("/")
            insts = glob.glob("sequences/" + names[-3] + "/instance/" + names[-1][:-4] + "_*.bin")
            insts_all = {}
            labels_all = {}

            '''
            for s_class in instance_classes:
                block_ = np.delete(block_, (labels_ == s_class), axis=0)
                labels_ = np.delete(labels_, (labels_ == s_class), axis=0)
            '''
            count = 0
            for inst_name in insts:
                if self.inst_path_label[inst_name] == 1:
                    continue
                inst_point = np.fromfile(inst_name, dtype=np.float32).reshape(-1, 4)
                #inst_point = self.twis(inst_point)
                insts_all[count] = inst_point
                la = np.ones(inst_point.shape[0]) * self.inst_path_label[inst_name] - 1
                labels_all[count] = la
                count += 1

            if count > 0:
                insts_paste = []
                labels_paste = []
                insts_tmp = {}
                labels_tmp = {}
                #print(block_.shape, labels_.shape, count, "* " * 10)
                #road_plane = block_[labels_==8][:,:3]
                #road_plane = np.clip(road_plane, np.random.randint(-40, -25), np.random.randint(25, 40))            
                road_plane = self.thin_road(block_, labels_) 


                sample_p = self.farthest_point_sample(road_plane, np.random.randint(count*2+1, count*4+2))
                iter_num = 0
                for center in sample_p:
                    instance_p = copy.deepcopy(insts_all[iter_num%count])
                    instance_l = labels_all[iter_num%count]
                    instance_p[:,:2] += (center[0,0,:2] - np.mean(instance_p, 0)[:2]) 
                    #instance_p = self.twis(instance_p)
                    #_, instance_xyz = self.pointwolf(instance_p[:,:3])
                    instance_p[:,:3] = instance_xyz
                    insts_paste.append(instance_p)
                    labels_paste.append(instance_l)
                    iter_num += 1
                    '''
                    if iter_num == count:
                        tmp_count = 0
                        for keys in insts_all:
                            if labels_all[keys][0] == 0:
                                pass
                                #print("* " * 10)
                            else:
                                insts_tmp[tmp_count] = insts_all[keys]
                                labels_tmp[tmp_count] = labels_all[keys]
                                tmp_count += 1

                        count = tmp_count
                        insts_all = insts_tmp
                        labels_all = labels_tmp
                        #print(tmp_count, "* " * 10, len(sample_p))
                        if tmp_count < 1:
                            break
                    '''

                #block_.astype(np.float32).tofile(str(self.count) + ".bin")
                #labels_.astype(np.uint32).tofile(str(self.count) + ".label") 
                #self.count += 1
                #print(len(insts_paste), len(insts_all), insts_paste[0].shape, "* " * 10) 
                if len(insts_all) > 0:
                    insts_paste = np.concatenate(insts_paste, axis=0)
                    labels_paste = np.concatenate(labels_paste, axis=0)

                    block_ = np.concatenate([block_, insts_paste], axis=0)
                    labels_ = np.concatenate([labels_, labels_paste], axis=0)

                block_ = self.twis_scene(block_)           
                #block_.astype(np.float32).tofile(str(self.count) + ".bin")
                #labels_.astype(np.uint32).tofile(str(self.count) + ".label")
                #self.count += 1
  
            '''
            # polarmix
            alpha = (np.random.random() - 1) * np.pi
            beta = alpha + np.pi
            block_, labels_ = polarmix(block_, labels_, pts2, labels2,
                                      alpha=alpha, beta=beta,
                                      instance_classes=instance_classes,
                                      Omega=Omega)            
            ''' 

        block = np.zeros_like(block_)

        if 'train' in self.split:
            theta = np.random.uniform(0, 2 * np.pi)
            scale_factor = np.random.uniform(0.95, 1.05)
            rot_mat = np.array([[np.cos(theta), np.sin(theta), 0],
                                [-np.sin(theta),
                                 np.cos(theta), 0], [0, 0, 1]])
            block[:, :3] = np.dot(block_[:, :3], rot_mat) * scale_factor
        else:
            theta = self.angle
            transform_mat = np.array([[np.cos(theta),
                                       np.sin(theta), 0],
                                      [-np.sin(theta),
                                       np.cos(theta), 0], [0, 0, 1]])
            block[...] = block_[...]
            block[:, :3] = np.dot(block[:, :3], transform_mat)

        block[:, 3] = block_[:, 3]
        pc_ = np.round(block[:, :3] / self.voxel_size).astype(np.int32)
        pc_ -= pc_.min(0, keepdims=1)

        feat_ = block

        _, inds, inverse_map = sparse_quantize(pc_,
                                               return_index=True,
                                               return_inverse=True)

        if 'train' in self.split:
            if len(inds) > self.num_points:
                inds = np.random.choice(inds, self.num_points, replace=False)

        pc = pc_[inds]
        feat = feat_[inds]
        labels = labels_[inds]
        lidar = SparseTensor(feat, pc)
        labels = SparseTensor(labels, pc)
        labels_ = SparseTensor(labels_, pc_)
        inverse_map = SparseTensor(inverse_map, pc_)

        return {
            'lidar': lidar,
            'targets': labels,
            'targets_mapped': labels_,
            'inverse_map': inverse_map,
            'file_name': self.files[index]
        }

    @staticmethod
    def collate_fn(inputs):
        return sparse_collate_fn(inputs)
