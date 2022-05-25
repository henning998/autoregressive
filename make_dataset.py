import time

import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch import Tensor, nn
from torch.nn import functional as F
import pyrender
import trimesh
from numpy.linalg import inv
import pytorch_lightning as pl
import random

import time
import threading


# Implementation of code from Troels' test:

# Map the 3d point to pixel value
def map_to_pixel(point_pose,w,h,projection,view):
    point3d = np.array([point_pose[0][3],point_pose[1][3],point_pose[2][3],1.])
    p=projection@inv(view)@point3d
    p=p/p[3]
    p[0]=round(w/2*p[0]+w/2)   #tranformation from [-1,1] ->[0,width]
    p[1]=round(h-(h/2*p[1]+h/2))  #tranformation from [-1,1] ->[0,height] (top-left image)
    
    return p.astype(int)

def random_transform_matrix():
  
  # x = random.uniform(0,90)
  # y = random.uniform(0,90)
  # z = random.uniform(0,90)
  rot = R.random().as_matrix()
  T = np.eye(4)
  translation = np.random.uniform(low = -1.7 , high = 1.7 , size = (3))
  translation[2] = 0
  T[:3,:3] = rot
  T[:3,3] = translation
  
  return T

def spherical_data_fun():
    # Generate the sphere and points:
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=1.0, color=None)
    sphere.visual.vertex_colors = [200, 0, 250, 100]
    sphere_mesh = pyrender.Mesh.from_trimesh(sphere)

    point = trimesh.creation.icosphere(subdivisions=3, radius=0.1, color=None)
    point1_mesh = pyrender.Mesh.from_trimesh(point)
    point2_mesh = pyrender.Mesh.from_trimesh(point)
    point3_mesh = pyrender.Mesh.from_trimesh(point)
    point4_mesh = pyrender.Mesh.from_trimesh(point)

    # Pose for the sphere and camera
    sphere_pose = np.array([[[1. , 0. , 0. , 0. ],
                        [0. , 1. , 0. , 0. ],
                        [0. , 0. , 1. , 0. ],
                        [0. , 0. , 0. , 1. ]]])

    cam_pose = np.array([[1. , 0. , 0. , 0.],
                        [0. , 1. , 0. , 0. ],
                        [0. , 0. , 1. , 5. ],
                        [0. , 0. , 0. , 1. ]])

    # Poses for the points
    point_poses = np.array([[[1. , 0. , 0. , 1.],
                        [0. , 1. , 0. , 0. ],
                        [0. , 0. , 1. , 0. ],
                        [0. , 0. , 0. , 1. ]],
                    [[1. , 0. , 0. , -1.],
                        [0. , 1. , 0. , 0. ],
                        [0. , 0. , 1. , 0. ],
                        [0. , 0. , 0. , 1. ]],
                    [[1. , 0. , 0. , 0.],
                        [0. , 1. , 0. , 1. ],
                        [0. , 0. , 1. , 0. ],
                        [0. , 0. , 0. , 1. ]],
                    [[1. , 0. , 0. , 0.],
                        [0. , 1. , 0. , -1. ],
                        [0. , 0. , 1. , 0. ],
                        [0. , 0. , 0. , 1. ]]])

    # Setup the perspective camera:
    cam = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)

    # Setup the scene and add the camera and the sphere+points with random rotation
    scene = pyrender.Scene()

    # Get the random transformation matrix:
    rand_trans = random_transform_matrix()

    scene.add(sphere_mesh, pose=sphere_pose[0]@rand_trans)
    # scene.add(point1_mesh, pose=rand_trans@point_poses[0])
    # scene.add(point2_mesh, pose=rand_trans@point_poses[1])
    # scene.add(point3_mesh, pose=rand_trans@point_poses[2])
    # scene.add(point4_mesh, pose=rand_trans@point_poses[3])
    scene.add(cam, pose=cam_pose, name='CAM')

    # Projection matrix
    pro_mat = cam.get_projection_matrix()

    # Setup the Off-screen Rendering
    #start = time.time()
    width = 128
    height = 128
    render = pyrender.offscreen.OffscreenRenderer(width, height) # width, height
    img, depth = render.render(scene)
    #print("data fun",time.time() - start)

    # Test how the image looks:
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # cv2_imshow(img)


    # Get the 2d pixel location from the points
    result1 = map_to_pixel(rand_trans@point_poses[0], width, height, pro_mat, cam_pose)
    result2 = map_to_pixel(rand_trans@point_poses[1], width, height, pro_mat, cam_pose)
    result3 = map_to_pixel(rand_trans@point_poses[2], width, height, pro_mat, cam_pose)
    result4 = map_to_pixel(rand_trans@point_poses[3], width, height, pro_mat, cam_pose)
    h = height
    w = width
  
  

    pts = np.array([result1[:2],result2[:2],result3[:2],result4[:2]]).reshape(4,2)
    temp = np.zeros((h,w),dtype=np.uint8)
    temp[int(pts[0][1]),int(pts[0][0])]=1
    img = np.append(img,temp)
    img = img.reshape(2,h,w)
    temp = np.zeros((h,w),dtype=np.uint8)
    temp[int(pts[1][1]),int(pts[1][0])]=1
    img = np.append(img,temp)
    img = img.reshape(3,h,w)
    temp = np.zeros((h,w),dtype=np.uint8)
    temp[int(pts[2][1]),int(pts[2][0])]=1
    img = np.append(img,temp)
    img = img.reshape(4,h,w)
    temp = np.zeros((h,w),dtype=np.uint8)
    temp[int(pts[3][1]),int(pts[3][0])]=1
    img = np.append(img,temp)
    img = img.reshape(5,h,w)


    return img

for i in tqdm(range(0,500)):
    testimg = spherical_data_fun()
    #cv2.imshow("test",testimg[0])
    #cv2.waitKey(0)
    with open('dataset/' + str(i) + '.npy', 'wb') as f:
        np.save(f, testimg)


with open('dataset/' + str(i) + '.npy', 'rb') as f:
    a = np.load(f)

# cv2.imshow("numpy",a[0])
# cv2.waitKey(0)