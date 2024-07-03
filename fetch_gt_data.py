import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import sklearn
import yaml
#from addict import Dict
import pandas as pd
#import plyfile
import tqdm
import scipy
import open3d as o3d
from itertools import islice, cycle
import pickle as pkl
#from sklearn.cluster import DBSCAN
#from sklearn.cluster import Birch
#import keyboard
import os

#import matlab.engine

#---------------------------------------------------------------------------

def fetch_gt_data(seq_num_str):
	#seq_num_str = "00"

	gt_poses_file_path = "./kitti_dataset/dataset/poses/{}.txt".format(seq_num_str)

	# read in poses from gt file (camera-to-world matrices from kitti)
	gt_pose_df = pd.read_csv(
		gt_poses_file_path,
		sep=" ", names=[
			"pose_11","pose_12","pose_13","pose_14",
			"pose_21","pose_22","pose_23","pose_24",
			"pose_31","pose_32","pose_33","pose_34"
		], dtype={
			"pose_11": "float64","pose_12": "float64","pose_13": "float64","pose_14": "float64",
			"pose_21": "float64","pose_22": "float64","pose_23": "float64","pose_24": "float64",
			"pose_31": "float64","pose_32": "float64","pose_33": "float64","pose_34": "float64"
		}, header=None, index_col=False)
	#print(gt_pose_df.head(4))

	gt_dict = {"kitti_xyz":[], "kitti_c2w":[]}
	for index, kitti_row in gt_pose_df.iterrows():
		kitti_xyz = np.asarray([kitti_row["pose_14"],kitti_row["pose_24"],kitti_row["pose_34"]])
		
		kitti_c2w = np.asarray([
			[kitti_row["pose_11"], kitti_row["pose_12"], kitti_row["pose_13"], kitti_row["pose_14"]],
			[kitti_row["pose_21"], kitti_row["pose_22"], kitti_row["pose_23"], kitti_row["pose_24"]],
			[kitti_row["pose_31"], kitti_row["pose_32"], kitti_row["pose_33"], kitti_row["pose_34"]],
			[0.0,0.0,0.0,0.1],
		])
		
		gt_dict["kitti_xyz"].append(kitti_xyz)
		gt_dict["kitti_c2w"].append(kitti_c2w)
	
	return gt_dict



if __name__ == "__main__":
	seq_num = "00"
	
	testobj = fetch_gt_data(seq_num)

















