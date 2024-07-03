import time
import numpy as np
import matplotlib.pyplot as plt
#import sklearn
#import yaml
##from addict import Dict
import pandas as pd
##import plyfile
#import tqdm
#import scipy
#import open3d as o3d

import pykitti
import cv2
import open3d as o3d

import copy

#import matlab.engine

np.set_printoptions(edgeitems=10,linewidth=180)

CITYSCAPES_CATEGORIES = [
    {"color": (128, 64, 128), "isthing": 0, "id": 7, "trainId": 0, "name": "road"},
    {"color": (244, 35, 232), "isthing": 0, "id": 8, "trainId": 1, "name": "sidewalk"},
    {"color": (70, 70, 70), "isthing": 0, "id": 11, "trainId": 2, "name": "building"},
    {"color": (102, 102, 156), "isthing": 0, "id": 12, "trainId": 3, "name": "wall"},
    {"color": (190, 153, 153), "isthing": 0, "id": 13, "trainId": 4, "name": "fence"},
    {"color": (153, 153, 153), "isthing": 0, "id": 17, "trainId": 5, "name": "pole"},
    {"color": (250, 170, 30), "isthing": 0, "id": 19, "trainId": 6, "name": "traffic light"},
    {"color": (220, 220, 0), "isthing": 0, "id": 20, "trainId": 7, "name": "traffic sign"},
    {"color": (107, 142, 35), "isthing": 0, "id": 21, "trainId": 8, "name": "vegetation"},
    {"color": (152, 251, 152), "isthing": 0, "id": 22, "trainId": 9, "name": "terrain"},
    {"color": (70, 130, 180), "isthing": 0, "id": 23, "trainId": 10, "name": "sky"},
    {"color": (220, 20, 60), "isthing": 1, "id": 24, "trainId": 11, "name": "person"},
    {"color": (255, 0, 0), "isthing": 1, "id": 25, "trainId": 12, "name": "rider"},
    {"color": (0, 0, 142), "isthing": 1, "id": 26, "trainId": 13, "name": "car"},
    {"color": (0, 0, 70), "isthing": 1, "id": 27, "trainId": 14, "name": "truck"},
    {"color": (0, 60, 100), "isthing": 1, "id": 28, "trainId": 15, "name": "bus"},
    {"color": (0, 80, 100), "isthing": 1, "id": 31, "trainId": 16, "name": "train"},
    {"color": (0, 0, 230), "isthing": 1, "id": 32, "trainId": 17, "name": "motorcycle"},
    {"color": (119, 11, 32), "isthing": 1, "id": 33, "trainId": 18, "name": "bicycle"}
]

#---------------------------------------------------------------------------

# returns poses, points belonging to each pose
# returns frame number, w2c, c2w, xyz position, array of points
def fetch_sequence_data(poses_file_path, points_file_path, NUM_SEARCH_FRAMES, SCAN_RADIUS, MAXIMUM_ACCURATE_RANGE, seq_num="00", kitti_dataset_path = "/bulk-pool/archive/kitti_odometry_assemble/dataset", all_mask_data_path = "./semantic_label_data"):
	
	#---------------------------------------------------------------------------
	# -for each frame ID:
	#	-w2c matrix
	#	-c2w matrix
	#	-XYZ camera position
	#	-few points (XYZ) associated with this frame
	# *		-the uv coords of those points
	# *		-their labels
	#	-collected points into scan (within range)
	# *		-the uv, labels for those
	#	
	
	scan_dict = {
		"frame_id" : [],		# list, one element per frame, each element is an int (scan's corresponding frame number)
		"w2c_matrix" : [],		# initial transformation matrix from DSO, world to camera
		"c2w_matrix" : [],		# transformation from camera to world coords, where the points are
		"frame_pose" : [],		# the XYZ position of the camera center in world coords
		"frame_points" : [],	# list, one element per frame, each element is a numpy (n,3) array with n points from the root frame
		"frame_2d_pos" : [],	# the 2D image positions for each of those points, u and v (x and y)
		"frame_labels" : [],	# the semantic labels of those points
		"scan_points" : [],		# list, one element per frame, each element is a numpy (n,3) array with n accumulated scan points
		"scan_2d_pos" : [],		# the 2D image positions for each of the scan points, u and v (x and y) and the kitti frame ID they were found in
		"scan_labels" : []		# the semantic labels of the scan points
	}
	
	# read in poses
	pose_df = pd.read_csv(
		poses_file_path,
		sep=" ", names=[
			"incoming_id",
			"pose_11","pose_12","pose_13","pose_14",
			"pose_21","pose_22","pose_23","pose_24",
			"pose_31","pose_32","pose_33","pose_34"
		], dtype={
			"incoming_id": "int",
			"pose_11": "float64","pose_12": "float64","pose_13": "float64","pose_14": "float64",
			"pose_21": "float64","pose_22": "float64","pose_23": "float64","pose_24": "float64",
			"pose_31": "float64","pose_32": "float64","pose_33": "float64","pose_34": "float64"
		}, header=None, index_col=False
	)

	# read in their points
	point_df = pd.read_csv(
		points_file_path, sep=" ", names=[
			"incoming_id","x","y","z","intensity"
		], dtype={
			"incoming_id": "int",
			"x": "float64",
			"y": "float64",
			"z": "float64",
			"intensity": "float64"
		}, header=None, index_col=False
	)
	
	#---------------------------------------------------------------------------
	# -for each frame ID:
	#	-w2c matrix
	#	-c2w matrix
	#	-XYZ camera position
	#	-points (XYZ) associated with this frame
	# *		-the uv coords of those points
	# *		-their labels
	#	-collected points into scan (within range)
	# *		-the uv, labels for those
	#	
	
	# process each frame from the list of poses (these are the frames we have data for)
	for index, row in pose_df.iterrows():
		#----------------------------------------------------------------------------------
		# get frame_id and points belonging to this frame
		
		# get the frame's ID
		frame_id = int(row["incoming_id"])
		
		# get this frame's points in world coordinates
		this_frame_points = point_df.loc[point_df['incoming_id'] == frame_id][["x", "y", "z"]].to_numpy()
		
		
		scan_dict["frame_id"].append(frame_id)
		#----------------------------------------------------------------------------------
		# get w2c matrix, c2w matrix, and frame XYZ position
		
		# the poses are expressed as world-to-camera matrices
		# we need the camera-to-world matrices to extract the position of the camera center from
		
		w2c_matrix = np.asarray([
			[row["pose_11"],row["pose_12"],row["pose_13"],row["pose_14"]],
			[row["pose_21"],row["pose_22"],row["pose_23"],row["pose_24"]],
			[row["pose_31"],row["pose_32"],row["pose_33"],row["pose_34"]],
			[0.0,			0.0,			0.0,			1.0]
		])
		w2c_r = np.asarray([
			[row["pose_11"],row["pose_12"],row["pose_13"]],
			[row["pose_21"],row["pose_22"],row["pose_23"]],
			[row["pose_31"],row["pose_32"],row["pose_33"]]
		])
		w2c_t = np.asarray([
			[row["pose_14"]],
			[row["pose_24"]],
			[row["pose_34"]],
		])
		
		# find c2w_r and c2w_t
		c2w_r = np.transpose(w2c_r)			#R'
		c2w_t = -1*np.matmul(c2w_r,w2c_t)	#-R't
		
		# find c2w_matrix
		c2w_matrix = np.hstack((c2w_r,c2w_t))
		c2w_matrix = np.vstack((c2w_matrix,np.asarray([0.0,0.0,0.0,1.0])))
		
		# compute camera location in world
		camera_center_pos = np.matmul(c2w_matrix,np.asarray([[0.0],[0.0],[0.0],[1.0]]))
		frame_xyz = np.asarray([camera_center_pos[0,0],camera_center_pos[1,0],camera_center_pos[2,0]])
		
		
		scan_dict["w2c_matrix"].append(w2c_matrix)
		scan_dict["c2w_matrix"].append(c2w_matrix)
		scan_dict["frame_pose"].append(frame_xyz)
		#----------------------------------------------------------------------------------
		# taking this frame's points and filtering them based on the frame position just calculated
		
		#if this_frame_points.shape[0] == 0:
		#	print("frame {} has no points associated with it!".format(frame_id))
		#frame 1886 has no points associated with it!
		
		# clip each frame's contributed points as being sufficiently close to the camera for accuracy
		# clip this_frame_points based on frame_xyz and MAXIMUM_ACCURATE_RANGE
		dists = frame_xyz - this_frame_points
		dists = np.square(dists)
		dists = np.sum(dists, axis=1)
		dists = np.sqrt(dists)
		mask = dists < MAXIMUM_ACCURATE_RANGE
		this_frame_points = this_frame_points[mask]
		
		
		scan_dict["frame_points"].append(this_frame_points)
		#----------------------------------------------------------------------------------
		# use calibrarion info to find these points' 2D image locations
		
		#kitti_dataset_path = "/bulk-pool/archive/kitti_odometry_assemble/dataset"
		#sequence_num = '00'
		
		# dataset.calib:      Calibration data are accessible as a named tuple
		# dataset.timestamps: Timestamps are parsed into a list of timedelta objects
		# dataset.poses:      List of ground truth poses T_w_cam0
		# dataset.camN:       Generator to load individual images from camera N
		# dataset.gray:       Generator to load monochrome stereo pairs (cam0, cam1)
		# dataset.rgb:        Generator to load RGB stereo pairs (cam2, cam3)
		# dataset.velo:       Generator to load velodyne scans as [x,y,z,reflectance]
		
		dataset_obj = pykitti.odometry(kitti_dataset_path, seq_num)
		
		# seq 00 default
		#P_rect_00 = np.asarray([
		#	[718.856,	0.0,		607.1928,	0.0],
		#	[0.0,		718.856,	185.2157,	0.0],
		#	[0.0,		0.0,		1.0,		0.0]
		#])
		
		P_rect_00 = dataset_obj.calib.P_rect_00
		#R_rect_00 = dataset_obj.calib.R_rect_00		# doesn't exist for odometry dataset
		#T_cam0_velo = dataset_obj.calib.T_cam0_velo
		
		'''
		# hacky improvement attempt for poor calibration (seq 00)
		if seq_num == "00":
			#P_rect_00 = np.asarray([
			#	[675.0,		0.0,	590.1928,	0.0],
			#	[0.0,		675.0,	185.2157,	0.0],
			#	[0.0,		0.0,	1.0,		0.0]
			#])
			P_rect_00 = np.asarray([
				[695.0,		0.0,	610.0,	0.0],
				[0.0,		695.0,	185.2157,	0.0],
				[0.0,		0.0,	1.0,		0.0]
			])
		'''
		
		#print("P_rect_00:\n{}".format(P_rect_00))
		#print("this_frame_points: {}".format(this_frame_points.shape))
		
		# transform points(?), convert points to camera coords, transform back(?)
		
		this_frame_points_h = np.concatenate((this_frame_points, np.ones((this_frame_points.shape[0],1))), axis=1)
		#print("this_frame_points_h: {}".format(this_frame_points_h.shape))
		this_frame_points_h_t = np.transpose(this_frame_points_h)
		#print("this_frame_points_h_t: {}".format(this_frame_points_h_t.shape))
		cam_space_points_h_t = np.matmul(w2c_matrix,this_frame_points_h_t)
		#print("cam_space_points_h_t: {}".format(cam_space_points_h_t.shape))
		
		# transform points(?), project to 2D coords, transform back(?)
			# may or may not have to clip Z (all points should already be in front of the camera)
		
		# points must be in front of the camera (positive Z component)
		mask = cam_space_points_h_t[2, :] >= 0
		#print("mask: {}".format(mask.shape))
		cam_space_points_h_t = cam_space_points_h_t[:, mask]
		#print("cam_space_points_h_t: {}".format(cam_space_points_h_t.shape))
		
		# perform the projection from 3D to 2D and division by z coord
		points_2d_h_t = P_rect_00.dot(cam_space_points_h_t)
		#print("points_2d_h_t: {}".format(points_2d_h_t.shape))
		points_2d_h_t /= points_2d_h_t[2,:]
		
		this_frame_points_2d = np.transpose(points_2d_h_t[0:2,:])
		
		#print("points_2d_h_t: {}".format(points_2d_h_t.shape))
		#print("points_2d_h_t:\n{}".format(points_2d_h_t[:,0:9]))
		
		#print("this_frame_points.shape: {}".format(this_frame_points.shape))
		#print("this_frame_points_2d.shape: {}".format(this_frame_points_2d.shape))
		
		# here for reference only
		'''
		scanpts1_h = np.concatenate((scan_points1, np.ones((scan_points1.shape[0],1))), axis=1) # -> Nx3 was transposed
		scanpts1_cam_h = np.matmul(w2c_matrix1,np.transpose(scanpts1_h))
		scanpts1_global_h = np.matmul(c2w_kitti1,scanpts1_cam_h)
		scanpts1_global = np.transpose(scanpts1_global_h)[:,0:3]
		'''
		'''
		velo_temp = frame_velo.transpose()	# convert to a row of column vectors
		velo_temp [3,:] = 1					# convert to homogenious coords (last elelment used to be reflectance, now repurposed)
		mask = velo_temp[2, :] >= 0			# only take points that are in front of the camera
		velo_temp = velo_temp[:, mask]
		
		# perform the projection from 3D to 2D
		# 2D_point = P_rect_00 * T_cam0_velo * 3D_velo_point
		proj_points = P_rect_00.dot(T_cam0_velo.dot(velo_temp))
		proj_points /= proj_points[2,:]
		'''
		
		scan_dict["frame_2d_pos"].append(this_frame_points_2d)
		#----------------------------------------------------------------------------------
		
		mask_data_path = "{}/seq{}_semantic_mask_output_cityscape_key/".format(all_mask_data_path,seq_num)
		
		# load image or stored image masks
		frame_string = frame_string = f"{frame_id:06}.png.npy"
		frame_path = "{}{}".format(mask_data_path,frame_string)
		#print(frame_path)
		
		semantic_mask_data = np.load(frame_path)
		
		# points must be inside the bounds of the image
		# already checked that they're in front of the camera above (positive Z component in camera coords)
		temp_points_2d = this_frame_points_2d.copy()
		mask = (temp_points_2d[:, 0] < semantic_mask_data.shape[1]) & \
			(temp_points_2d[:, 0] >= 0) & \
			(temp_points_2d[:, 1] < semantic_mask_data.shape[0]) & \
			(temp_points_2d[:, 1] >= 0)
		temp_points_2d = temp_points_2d[mask, :]
		
		point_semantic_labels = []
		for j in range(temp_points_2d.shape[0]):
			x = int(temp_points_2d[j,0])
			y = int(temp_points_2d[j,1])
			
			cityscapes_id = semantic_mask_data[y,x] #y,x
			
			point_semantic_labels.append(cityscapes_id)
		point_semantic_labels = np.asarray(point_semantic_labels, dtype=np.int64)
		
		'''
		#TODO
		if temp_points_2d.shape[0] == 0:
			print("temp_points_2d.shape: {}".format(temp_points_2d.shape))
			print("point_semantic_labels.shape: {}".format(point_semantic_labels.shape))
			print("id: {}".format(frame_id))
			print()
		
		#TODO
		if point_semantic_labels.dtype != np.int64:
			print("point_semantic_labels.dtype: {}".format(point_semantic_labels.dtype))
			print("point_semantic_labels.shape: {}".format(point_semantic_labels.shape))
			print("id: {}".format(frame_id))
			print("temp_points_2d.shape: {}".format(temp_points_2d.shape))
			print(temp_points_2d.shape[0])
			print()
		'''
		
		#print("this_frame_points_2d.shape: {}".format(this_frame_points_2d.shape))
		#print("point_semantic_labels.shape: {}".format(point_semantic_labels.shape))
		
		scan_dict["frame_labels"].append(point_semantic_labels)
		#----------------------------------------------------------------------------------
		# collect the points of several frames together into a scan
		# also collect their 2D positions (and host frame IDs)
		
		# testing
		#flag = False #TODO
		
		min_frame_to_check = max(0, frame_id - NUM_SEARCH_FRAMES)
		point_accumulator = np.empty((0,3)) # np accumulator for this scan's points
		point_accumulator_2d = np.empty((0,3)) # np accumulator for this scan's points (u,v,frame_id)
		label_accumulator_2d = np.empty((0), dtype=np.int64)
		for j in reversed(range(len(scan_dict["frame_id"]))):
			current_frame_id = scan_dict["frame_id"][j]
			#print("j: {}, current_frame_id:{}".format(j,current_frame_id))
			
			# i indexes over the already-processed DSO frames, including the current one
			# (note that some fields, most notably the scan-related ones produced by this accumulator, won't be commited yet and can't be used)
			if not (current_frame_id >= min_frame_to_check):
				# if the current frame's ID is below the current minimum frame (backwards in time) to check, skip it
				# we don't need an upper bound because we can't process frames in the future, only up to the current frame
				#continue
				break #TODO # we count downwards, so in theory this breaks on the first one we don't care about, just below range?
			
			#-------------------------------------------------------------------
			
			# slightly different point gathering behavior compared with below?
			frame_points = scan_dict["frame_points"][j]
			frame_points_2d = scan_dict["frame_2d_pos"][j]
			frame_labels_2d = scan_dict["frame_labels"][j]
			
			#print("number of frame points: {}".format(frame_points.shape)) #shape[0] would suffice
			#print("number of frame points 2d: {}".format(frame_points_2d.shape)) #shape[0] would suffice
			
			# find distances between points and frame's camera center
			# require that they be within the desired pseudo-pointcloud's radius
			dists = frame_xyz - frame_points # frame_xyz of current frame
			dists = np.square(dists)
			dists = np.sum(dists, axis=1)
			dists = np.sqrt(dists)
			mask = dists < SCAN_RADIUS
			close_points = frame_points[mask]
			close_points_2d = frame_points_2d[mask]
			close_labels_2d = frame_labels_2d[mask]
			#print("number of close points: {}".format(close_points.shape))
			#print("number of close points 2d: {}".format(close_points_2d.shape))
			
			# add an extra column with the id of the frame these points came from
			#print("close_points_2d:\n{}".format(close_points_2d[0:9,:]))
			id_column_vec = np.full((close_points_2d.shape[0], 1), current_frame_id)
			close_points_2d = np.append(close_points_2d, id_column_vec, axis=1)
			#print("close_points_2d:\n{}".format(close_points_2d[0:9,:]))
			
			'''
			#TODO
			if close_labels_2d.dtype != np.int64:
				print("point_semantic_labels.dtype: {}".format(point_semantic_labels.dtype))
				print("point_semantic_labels.shape: {}".format(point_semantic_labels.shape))
				print("temp_points_2d.shape: {}".format(temp_points_2d.shape))
				print("-----")
				print("close_labels_2d.dtype != np.int64 -> {} val: {}".format(close_labels_2d.dtype,close_labels_2d))				#TODO track down the source of this bug
				print("mask.shape: {}".format(mask.shape))
				print("close_labels_2d.shape: {}".format(close_labels_2d.shape))
				print("dists.shape: {}".format(dists.shape))
				print("frame_labels_2d.shape: {}".format(frame_labels_2d.shape))
				print("frame_points_2d.shape: {}".format(frame_points_2d.shape))
				print("frame_points.shape: {}".format(frame_points.shape))
				print("frame_labels_2d.dtype: {}".format(frame_labels_2d.dtype))
				print("id: {}".format(frame_id))
				print()
			'''
			
			# add collected points from this frame to the accumulated point cloud
			point_accumulator = np.append(point_accumulator, close_points, axis=0)	
			point_accumulator_2d = np.append(point_accumulator_2d, close_points_2d, axis=0)
			label_accumulator_2d = np.append(label_accumulator_2d, close_labels_2d, axis=0)
			
			##TODO
			#if min_frame_to_check > 200:
			#	flag = True
				#print("frame_id: {} looking at points from: {}".format(index, j))
				#print("frame_xyz: {} frame_points.shape: {} dists.shape: {} mask.shape: {} close_points.shape: {}".format(frame_xyz,frame_points.shape,dists.shape,mask.shape,close_points.shape))
				#print("point_accumulator.shape: {}".format(point_accumulator.shape))
				#print("this_frame_points.shape: {} frame_points.shape: {}".format(this_frame_points.shape,frame_points.shape))
				#print()
		
		#TODO
		#if flag == True:
		#	#assert(0)
		#	pass
		
		'''
		point_accumulator = np.empty((0,3)) # np accumulator for this scan's points
		point_2d_accumulator = np.empty((0,2)) # np accumulator for this scan's points
		for j in range(frame_id, max(0, frame_id - NUM_SEARCH_FRAMES)-1, -1):
			if j in pose_df["incoming_id"].values:
				# for each frame at offsets from 0 to the maximum number of frames to search backwards in time (j, where j is among frame IDs)
				# if that frame actually exists in the list of frames (isn't a negative offset and was kept by the slam as a keyframe)
				# then we want to find the points which are within the desired radius of the current frame's camera center
				
				# get those points which belong to this frame
				frame_points = point_df.loc[point_df['incoming_id'] == j][["x", "y", "z"]].to_numpy()
				#print("number of frame points: {}".format(frame_points.shape))
				
				# find distances between points and frame's camera center
				# require that they be within the desired pseudo-pointcloud's radius
				dists = frame_xyz - frame_points # frame_xyz of current frame
				dists = np.square(dists)
				dists = np.sum(dists, axis=1)
				dists = np.sqrt(dists)
				mask = dists < SCAN_RADIUS
				close_points = frame_points[mask]
				print("number of close points: {}".format(close_points.shape))
				
				# add collected points from this frame to the accumulated point cloud
				point_accumulator = np.append(point_accumulator, close_points, axis=0)
		'''
		#print("frame {}'s shapes: {} points in this frame alone, {} total accumulated in it's associated point cloud".format(frame_id, this_frame_points.shape, point_accumulator.shape))
		#scan_dict["frame_id"].append(frame_id)
		#scan_dict["w2c_matrix"].append(w2c_matrix)
		#scan_dict["c2w_matrix"].append(c2w_matrix)
		#scan_dict["frame_pose"].append(frame_xyz)
		#scan_dict["frame_points"].append(this_frame_points) # probably useless, but used internally (critically!) by this script
		#scan_dict["frame_2d_pos"].append(this_frame_points_2d)
		#scan_dict["scan_points"].append(point_accumulator)
		# tbh frame_points is probably useless. if you want to do projection
		# it's better to do that while you're accumulating them so you can get
		# all the ones for this frame. using the focal length and w2c matrices
		# you should be able to project them all later anyway. (and check that
		# they're inside the viewing window using the resolution)
		
		
		scan_dict["scan_points"].append(point_accumulator)
		scan_dict["scan_2d_pos"].append(point_accumulator_2d) #TODO what is going on here!? what is it used for?
		scan_dict["scan_labels"].append(label_accumulator_2d)
		#----------------------------------------------------------------------------------
		# visualize every 100th scan
		'''
		if (not (index % 100)) and (index > 0):
			pcd = o3d.geometry.PointCloud()
			#pcd.points = o3d.utility.Vector3dVector(this_frame_points)
			pcd.points = o3d.utility.Vector3dVector(point_accumulator)
			#coord = o3d.geometry.TriangleMesh().create_coordinate_frame(size=20.0)
			coord1 = o3d.geometry.TriangleMesh().create_coordinate_frame(size=10.0,origin=frame_xyz)
			o3d.visualization.draw_geometries([pcd,coord1])
		'''
		
		'''
		frame_id: 135
		frame_id: 283
		frame_id: 416
		frame_id: 609
		frame_id: 729
		frame_id: 872
		frame_id: 996
		frame_id: 1109
		frame_id: 1272
		frame_id: 1437
		frame_id: 1549
		frame_id: 1674
		frame_id: 1807
		frame_id: 1919
		temp_label.dtype != np.int64 -> float64 val: 13.0
		'''
		
		'''
		close_labels_2d.dtype != np.int64 -> float64 val: []
		close_labels_2d.dtype != np.int64 -> float64 val: []
		close_labels_2d.dtype != np.int64 -> float64 val: []
		close_labels_2d.dtype != np.int64 -> float64 val: []
		frame_id: 2679
		'''
		
		# show semantically colored pseudo pointclouds
		'''
		if (not (index % 100)) and (index > 0):
			# load image by id
			frame_gray_0 = np.asarray(dataset_obj.get_gray(frame_id)[0])
			print("frame_id: {}".format(frame_id))
			
			#point_accumulator
			#label_accumulator_2d
			
			#print("point_accumulator.shape: {}".format(point_accumulator.shape))
			#print("label_accumulator_2d.shape: {}".format(label_accumulator_2d.shape))
			
			colors = []
			for i in range(point_accumulator.shape[0]):
				temp_label = label_accumulator_2d[i]
				
				if temp_label == -1:
					temp_color = (102, 102, 156)
				else:
					#if temp_label.dtype != np.int64:
					#	print("temp_label.dtype != np.int64 -> {} val: {}".format(temp_label.dtype,temp_label))				#TO DO track down the source of this bug
					temp_color = CITYSCAPES_CATEGORIES[temp_label]["color"]											#TO DO temporary int() fix
					assert CITYSCAPES_CATEGORIES[temp_label]["trainId"] == temp_label										# was doing an asarray() in the per-frame points without giving dtype, and it inferred wrong when there were zero points
				colors.append(temp_color)
			
			colors = np.asarray(colors)/255.0
			
			pcd = o3d.geometry.PointCloud()
			pcd.points = o3d.utility.Vector3dVector(point_accumulator)
			pcd.colors = o3d.utility.Vector3dVector(colors)
			coord1 = o3d.geometry.TriangleMesh().create_coordinate_frame(size=10.0,origin=frame_xyz)
			o3d.visualization.draw_geometries([pcd,coord1])
		'''
		
		# show semantically colored projected points
		'''
		if (not (index % 100)) and (index > 1900):
			# load image by id
			frame_gray_0 = np.asarray(dataset_obj.get_gray(frame_id)[0])
			print("frame_id: {}".format(frame_id))
			
			#point_accumulator = this_frame_points
			
			# move points to camera coords
			# w2c_matrix and P_rect_00 already found, constant scross sequence
			#print("point_accumulator.shape: {}".format(point_accumulator.shape))
			scan_points_h = np.concatenate((point_accumulator, np.ones((point_accumulator.shape[0],1))), axis=1)
			#print("scan_points_h.shape: {}".format(scan_points_h.shape))
			scan_points_h_t = np.transpose(scan_points_h)
			#print("scan_points_h_t.shape: {}".format(scan_points_h_t.shape))
			cam_space_points_h_t = np.matmul(w2c_matrix,scan_points_h_t)
			#print("cam_space_points_h_t: {}".format(cam_space_points_h_t.shape))
			
			# do Z clipping
			mask = cam_space_points_h_t[2, :] >= 0 					# test Z coordinate
			cam_space_points_h_t = cam_space_points_h_t[:, mask]	# only take points that are in front of the camera
			
			# fetch point labels and perform same z clip operation to keep the points and labels in sync
			scan_labels = label_accumulator_2d
			scan_labels_h = scan_labels.transpose()
			scan_labels_h = scan_labels_h[mask]
			
			# project from 3D to 2D and divide by z coord
			points_2d_h_t = P_rect_00.dot(cam_space_points_h_t)
			points_2d_h_t /= points_2d_h_t[2,:]
			
			#this_frame_points_2d = np.transpose(points_2d_h_t[0:2,:]) # Take only u and v (discard 1) and convert to vertical list of row vectors
			proj_points = points_2d_h_t
			
			# clip the projected points to within the image's viewing area
			# frame_gray_0.shape[1]	=> width
			# proj_points[0, :]		=> x coords
			# frame_gray_0.shape[0]	=> height
			# proj_points[0, :]		=> ycoords
			mask = (proj_points[0, :] < frame_gray_0.shape[1]) & \
					(proj_points[0, :] >= 0) & \
					(proj_points[1, :] < frame_gray_0.shape[0]) & \
					(proj_points[1, :] >= 0)
			proj_points = proj_points[:, mask]

			# perform same imabe bounds clip operation to keep the points and labels in sync
			scan_labels_h = scan_labels_h[mask]
			
			# draw on image
			#dot_image = np.copy(frame_gray_0)
			#dot_image = np.ones_like(frame_gray_0)
			#dot_image = np.stack((frame_gray_0,frame_gray_0,frame_gray_0), -1)
			dot_image = np.stack((np.ones_like(frame_gray_0),np.ones_like(frame_gray_0),np.ones_like(frame_gray_0)), -1)
			proj_points = proj_points.transpose()
			for i in range(proj_points.shape[0]):
				temp_label = scan_labels_h[i]
				
				if temp_label == -1:
					temp_color = (0, 0, 0)
				else:
					temp_color = CITYSCAPES_CATEGORIES[temp_label]["color"]
					assert CITYSCAPES_CATEGORIES[temp_label]["trainId"] == temp_label
				
				dot_image = cv2.circle(dot_image,
					(
						int(proj_points[i,0]),
						int(proj_points[i,1])
					), 1, color=temp_color, thickness=-1
				) #BGR
			
			# display image(s)
			#plt.figure("raw")
			#plt.imshow(frame_gray_0, cmap="gray")
			plt.figure("dots")
			#plt.imshow(np.asarray(dot_image), cmap="gray")
			plt.imshow(np.asarray(dot_image))
			plt.show()
			#assert(0)
		'''
		
		'''
		if (not (index % 100)) and (index > 0):
			# load image by id
			frame_gray_0 = np.asarray(dataset_obj.get_gray(frame_id)[0])
			print("frame_id: {}".format(frame_id))
			
			#point_accumulator = this_frame_points
			
			# move points to camera coords
			# w2c_matrix and P_rect_00 already found, constant scross sequence
			#print("point_accumulator.shape: {}".format(point_accumulator.shape))
			scan_points_h = np.concatenate((point_accumulator, np.ones((point_accumulator.shape[0],1))), axis=1)
			#print("scan_points_h.shape: {}".format(scan_points_h.shape))
			scan_points_h_t = np.transpose(scan_points_h)
			#print("scan_points_h_t.shape: {}".format(scan_points_h_t.shape))
			cam_space_points_h_t = np.matmul(w2c_matrix,scan_points_h_t)
			#print("cam_space_points_h_t: {}".format(cam_space_points_h_t.shape))
			
			# do Z clipping
			mask = cam_space_points_h_t[2, :] >= 0 					# test Z coordinate
			cam_space_points_h_t = cam_space_points_h_t[:, mask]	# only take points that are in front of the camera
			
			# project from 3D to 2D and divide by z coord
			points_2d_h_t = P_rect_00.dot(cam_space_points_h_t)
			points_2d_h_t /= points_2d_h_t[2,:]
			
			#this_frame_points_2d = np.transpose(points_2d_h_t[0:2,:]) # Take only u and v (discard 1) and convert to vertical list of row vectors
			proj_points = points_2d_h_t
			
			# clip the projected points to within the image's viewing area
			# frame_gray_0.shape[1]	=> width
			# proj_points[0, :]		=> x coords
			# frame_gray_0.shape[0]	=> height
			# proj_points[0, :]		=> ycoords
			mask = (proj_points[0, :] < frame_gray_0.shape[1]) & \
					(proj_points[0, :] >= 0) & \
					(proj_points[1, :] < frame_gray_0.shape[0]) & \
					(proj_points[1, :] >= 0)
			proj_points = proj_points[:, mask]
			
			# draw on image
			dot_image = np.copy(frame_gray_0)
			#dot_image = np.ones_like(frame_gray_0)
			proj_points = proj_points.transpose()
			for i in range(proj_points.shape[0]):
				dot_image = cv2.circle(dot_image,
					(
						int(proj_points[i,0]),
						int(proj_points[i,1])
					), 1, color=(0, 0, 0), thickness=-1
				) #BGR
			
			# display image(s)
			#plt.figure("raw")
			#plt.imshow(frame_gray_0, cmap="gray")
			plt.figure("dots")
			plt.imshow(np.asarray(dot_image), cmap="gray")
			plt.show()
			#assert(0)
		
		#print("this_frame_points: {}".format(this_frame_points.shape))
		'''
		
		
		#----------------------------------------------------------------------------------
	#for key in scan_dict:
	#	print("scan_dict[\"{}\"] => {}".format(key,len(scan_dict[key])))
	#assert(0)
	return scan_dict

# kept in above function
'''
# returns points belonging to each pose's pseudo-scan
# parameters: 
def generate_scans_from_points(poses, points_in_each_scan, parameter_dict):
	
	return None #dummy
'''

# testing
if __name__ == "__main__":
	seq_num = "00"					#TODO pass in sequence number (does it actually matter for just the calibration data?)
	
	NUM_SEARCH_FRAMES = 100
	SCAN_RADIUS = 45
	MAXIMUM_ACCURATE_RANGE = SCAN_RADIUS
	
	poses_file_path = "./kitti_dso_files/results_files/seq_{}/poses_history_file.txt".format(seq_num)
	points_file_path = "./kitti_dso_files/results_files/seq_{}/pts_history_file.txt".format(seq_num)
	
	# used to fetch calibration info
	#kitti_dataset_path = "/home/matt/Documents/waterloo/spatial_vpr/kitti_odometry/dataset"		# has calib,grey,poses
	kitti_dataset_path = "/bulk-pool/archive/kitti_odometry_assemble/dataset"						# adds velodyne data, could add color RGB data
	
	all_mask_data_path = "./semantic_label_data"
	
	out_thing = fetch_sequence_data(poses_file_path, points_file_path, NUM_SEARCH_FRAMES, SCAN_RADIUS, MAXIMUM_ACCURATE_RANGE, seq_num, kitti_dataset_path, all_mask_data_path)
	
	# getting the number with each kind of label
	scorecard = np.zeros(shape=(20))
	for frame_label_set in out_thing["frame_labels"]:
		frame_label_set[frame_label_set == -1] = 19
		bins = np.bincount(frame_label_set.flatten(), minlength=20)
		scorecard += bins
	test_obj = copy.deepcopy(CITYSCAPES_CATEGORIES)
	test_obj.append({"color": (0, 0, 0), "isthing": 0, "id": 500, "trainId": 19, "name": "invalid"})
	for i in range(scorecard.shape[0]):
		print("index: {} name: {} count: {} ".format(i, test_obj[i]["name"], scorecard[i]))

























