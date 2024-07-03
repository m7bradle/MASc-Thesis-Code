import numpy as np
import open3d as o3d
import pandas as pd
import matplotlib.pyplot as plt

#import argparse
import tempfile
import os
import subprocess

import time

import pickle as pkl

np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=149) # 150 max

# The criteria for used to determine ground-truth matches are intended to be the same as used by https://arxiv.org/abs/1909.07267
# ie. it must be within ten meters and must have occured at least 100 frames apart

#from fetch_pointcloud_sequence import fetch_sequence_data

'''
# select gt file
gt_poses_file_path = "./kitti_dataset/dataset/poses/00.txt"
#OUTPUT_FILE_PATH = "./kitti_groundtruth_poses/gt_lists_10m_00.npy"
MATCHING_DISTANCE_LIMIT = 10 #meters
#MATCHING_DISTANCE_LIMIT = 16
FRAME_NUMBER_MASK_RANGE = 100 #frames (generally with these kitti sequences 100 frames = 10 seconds)
NO_LOCAL_MASK = False #ignore self-similar
'''
#---------------------------------------------------------------------------

def get_groundtruth(gt_poses_file_path, MATCHING_DISTANCE_LIMIT=10, FRAME_NUMBER_MASK_RANGE=100, NO_LOCAL_MASK=False):

	# read in poses from gt file
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

	kitti_coords = []
	for index, kitti_row in gt_pose_df.iterrows():
		kitti_xyz = np.asarray([kitti_row["pose_14"],kitti_row["pose_24"],kitti_row["pose_34"]])
		kitti_coords.append(kitti_xyz)

	#---------------------------------------------------------------------------

	# for each frame, determine elidgible matches
	close_frames = []
	close_frames_dists = []
	strict_frames = []
	strict_frames_dists = []
	for i in range(len(kitti_coords)):				# for every frame in the kitti dataset, we have it's XYZ ground truth whether DOS gives it or not
		current_pos = kitti_coords[i]
		
		perframe_close_frames = []
		perframe_close_frame_dists = []
		min_frame = -1
		min_dist = 999999999.0
		for j in range(len(kitti_coords)):
			testing_pos = kitti_coords[j]
			
			dist = np.linalg.norm(current_pos - testing_pos)
			frame_diff = np.abs(i-j)
			
			#if (dist < MATCHING_DISTANCE_LIMIT) and (frame_diff > FRAME_NUMBER_MASK_RANGE):
			#if (dist < MATCHING_DISTANCE_LIMIT):
			if (dist < MATCHING_DISTANCE_LIMIT) and (NO_LOCAL_MASK or (frame_diff > FRAME_NUMBER_MASK_RANGE)):
				# if this frame meets the gneral criteria for being a matching frame, store it
				perframe_close_frames.append(j)
				perframe_close_frame_dists.append(dist)
				
				# check if this is the closest frame so far, replace if so
				if dist < min_dist:
					min_frame = j
					min_dist = dist
					
		# store list of acceptable matches
		close_frames.append(perframe_close_frames) 
		close_frames_dists.append(perframe_close_frame_dists)
		# store closest match, as per DSO paper's method
		strict_frames.append(min_frame)
		strict_frames_dists.append(min_dist)

	'''
	for i in range(len(kitti_coords)):
		print("\n{} : best {} at {}\naceptable: {}\ndists: {}".format(i,strict_frames[i],strict_frames_dists[i],close_frames[i],close_frames_dists[i]))
		
		if i > 100:
			break
	'''
	# write output file(s)
	gt_dict = {
		"close_frames" : close_frames,					# all the frames (indexes) which are a valid match for this frame (within physical range, not self-similar, etc)
		"close_frames_dists" : close_frames_dists,		# the distances from each matching frame to the current frame
		"strict_frames" : strict_frames,				# the absolute closest frame (index) to the current frame
		"strict_frames_dists" : strict_frames_dists		# the absolute closest frame's distance
	}
	
	return gt_dict


if __name__ == "__main__":
	
	gt_poses_file_path = "./kitti_dataset/dataset/poses/00.txt"
	# returns: close_frames, close_frames_dists, strict_frames, strict_frames_dists
	gt_dict = get_groundtruth(gt_poses_file_path) # the indexes of this are the full set, no missing frames, so must be indexed by frame_id
	#pkl.dump(gt_dict, open("./numpy_saves/00_gt_10-100-false.pkl", "wb"))
	#gt_dict = pkl.load(open("./numpy_saves/00_gt_10-100-false.pkl", "rb"))
	
	'''
	# returns: frame_id, w2c_matrix, c2w_matrix, frame_pose, frame_points, scan_points
	out_dict = fetch_sequence_data(poses_file_path, points_file_path, NUM_SEARCH_FRAMES, SCAN_RADIUS, MAXIMUM_ACCURATE_RANGE)
	#pkl.dump(out_dict, open("./numpy_saves/00_scans_100-45-45.pkl", "wb"))
	#out_dict = pkl.load(open("./numpy_saves/00_scans_100-45-45.pkl", "rb"))
	
	for i in range(2500,3040,36): # 15 equally spaced frames 		# these values are indexed into the array of keyframes we get from DSO, which does not contain every frame in the sequence
		frame_id = out_dict["frame_id"][i]							# because of this, we have to recover the corresponding frame id in the full set
		
		#close_frames = gt_dict["close_frames"][frame_id]
		#print("dso index {}, frame id {}: {}".format(i,frame_id,close_frames))
		
		strict_frames = gt_dict["strict_frames"][frame_id]
		print("dso index {}, frame id {}: {}".format(i,frame_id,strict_frames))
	'''






