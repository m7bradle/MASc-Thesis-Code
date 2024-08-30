import time
import datetime
import math
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
from sklearn.cluster import DBSCAN
from sklearn.cluster import Birch
import keyboard
import sklearn.neighbors as neighbors

#import clipperpy
from scipy.spatial.transform import Rotation as R

#from pyntcloud import PyntCloud

import os
import matplotlib.pyplot as plt

import io # used to redirect (silence) matlab output

import transforms3d

from LineMesh import LineMesh # module file in working directory for lines using open3D, constructed from cylinders

import matlab.engine

from fetch_pointcloud_sequence import fetch_sequence_data
from points_clustering_helper import do_clustering, label_points, generate_cluster_points
from fetch_gt_data import fetch_gt_data

from get_groundtruth_kitti import get_groundtruth # added since it was missing, dunno who ever called this before

import sys
np.set_printoptions(threshold=sys.maxsize)

import argparse

def uniform_random_rotation_helper(x, rng):
	"""Apply a random rotation in 3D, with a distribution uniform over the
	sphere.
	Arguments:
		x: vector or set of vectors with dimension (n, 3), where n is the
			number of vectors
	Returns:
		Array of shape (n, 3) containing the randomly rotated vectors of x,
		about the mean coordinate of x.
	Algorithm taken from "Fast Random Rotation Matrices" (James Avro, 1992):
	https://doi.org/10.1016/B978-0-08-050755-2.50034-8

	implementation based on
	https://www.blopig.com/blog/2021/08/uniformly-sampled-3d-rotation-matrices/
	"""
	def generate_random_z_axis_rotation(rng):
		"""Generate random rotation matrix about the z axis."""
		R = np.eye(3)
		x1 = rng.random()
		R[0, 0] = R[1, 1] = np.cos(2 * np.pi * x1)
		R[0, 1] = -np.sin(2 * np.pi * x1)
		R[1, 0] = np.sin(2 * np.pi * x1)
		return R
	# There are two random variables in [0, 1) here (naming is same as paper)
	x2 = 2 * np.pi * rng.random()
	x3 = rng.random()
	# Rotation of all points around x axis using matrix
	R = generate_random_z_axis_rotation(rng)
	v = np.array([
		np.cos(x2) * np.sqrt(x3),
		np.sin(x2) * np.sqrt(x3),
		np.sqrt(1 - x3)
	])
	H = np.eye(3) - (2 * np.outer(v, v))
	M = -(H @ R)
	x = x.reshape((-1, 3))
	mean_coord = np.mean(x, axis=0)
	return ((x - mean_coord) @ M) + mean_coord @ M

def uniform_random_rotation_matrix_helper(rng):
	# modified version that generates a random rotation matrix and returns it, rather than rotates points

	"""Apply a random rotation in 3D, with a distribution uniform over the
	sphere.
	Arguments:
	x: vector or set of vectors with dimension (n, 3), where n is the
		number of vectors
	Returns:
	Array of shape (n, 3) containing the randomly rotated vectors of x,
	about the mean coordinate of x.
	Algorithm taken from "Fast Random Rotation Matrices" (James Avro, 1992):
	https://doi.org/10.1016/B978-0-08-050755-2.50034-8

	implementation based on
	https://www.blopig.com/blog/2021/08/uniformly-sampled-3d-rotation-matrices/
	"""
	def generate_random_z_axis_rotation(rng):
		"""Generate random rotation matrix about the z axis."""
		R = np.eye(3)
		x1 = rng.random()
		R[0, 0] = R[1, 1] = np.cos(2 * np.pi * x1)
		R[0, 1] = -np.sin(2 * np.pi * x1)
		R[1, 0] = np.sin(2 * np.pi * x1)
		return R
	# There are two random variables in [0, 1) here (naming is same as paper)
	x2 = 2 * np.pi * rng.random()
	x3 = rng.random()
	# Rotation of all points around x axis using matrix
	R = generate_random_z_axis_rotation(rng)
	v = np.array([
		np.cos(x2) * np.sqrt(x3),
		np.sin(x2) * np.sqrt(x3),
		np.sqrt(1 - x3)
	])
	H = np.eye(3) - (2 * np.outer(v, v))
	M = -(H @ R)
	#x = x.reshape((-1, 3))
	#mean_coord = np.mean(x, axis=0)
	#return ((x - mean_coord) @ M) + mean_coord @ M
	return M

def find_halfnormal_stddev(samples):
	squared_samples = np.square(samples)
	num_samples = samples.shape[0]
	result = np.sum(squared_samples, axis=0)
	result = np.divide(result, num_samples)
	result = np.sqrt(result)
	return result

#---------------------------------------------------------------------------

# Create the parser and add arguments
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--sequence', dest='sequence', help="the 2-digit kitti sequence number", default="00", type=str, required=False)
parser.add_argument('-f', '--force_regen_scans', dest='force_regen_scan_data', help="force regeneration of scan data", default=False, type=bool, required=False)
parser.add_argument('-g', '--force_regen_gt_matches', dest='force_regen_match_data', help="force regeneration of gt valid match data", default=False, type=bool, required=False)

parser.add_argument('-p', '--force_fullseq_pair_gen', dest='force_fullseq_pair_gen', help="force search of whole-sequence when finding pairs of frames to test", default=False, type=bool, required=False)
parser.add_argument('-m', '--mode', dest='operating_mode', help="which benchmark to run", default=None, type=str, required=False)
parser.add_argument('-n', '--skip_every_n', dest='skip_num_frames', help="sample every n frames when looking for revisits to compare frames on", default=10, type=int, required=False)

parser.add_argument('-v', '--skip_vis', dest='skip_vis', help="skip visualization", default=False, type=bool, required=False)
parser.add_argument('-r', '--skip_graphs', dest='skip_graphs', help="skip graph showing", default=False, type=bool, required=False)

parser.add_argument('-o', '--cluster_output', dest='cluster_output_folder_path', help="where to put collected stats for a method's clusters", default="./results_output/collected_cluster_stats/test_folder/", type=str, required=False)
parser.add_argument('-q', '--use_semantic', dest='use_semantic', help="use semantic label based approach", default=None, type=str, required=False)
parser.add_argument('-c', '--clustering_override', dest='which_clustering_override', help="override which apprach to clustering is used", default=None, type=str, required=False)
parser.add_argument('-t', '--cluster_target_num', dest='cluster_target_num', help="how many clusters should be targeted for, typically 120 or 20 depending on method", default=None, type=int, required=False)

parser.add_argument('-d', '--alignment_output', dest='alignment_output_folder_path', help="where to put collected stats for alignment success testing", default="./results_output/collected_alignment_stats/test_folder/", type=str, required=False)
parser.add_argument('-l', '--gen_pcent_ouliers', dest='gen_pcent_ouliers', help="percent landmark outliers to add for testing", default=None, type=float, required=False)
parser.add_argument('-e', '--std_dev_noise', dest='std_dev_noise', help="std deviation of the landmarks generated for testing", default=None, type=float, required=False)

parser.add_argument('-u', '--use_uncorrected_totals', dest='use_uncorrected_totals', help="don't correct the total number of clusters found to be consistent across different scenes. may not be wired at the top level up for all modes.",
							 default=False, type=bool, required=False)

args = parser.parse_args()

#print(args.sequence)
#assert(0)
#---------------------------------------------------------------------------
NUM_SEARCH_FRAMES = 100
SCAN_RADIUS = 45
MAXIMUM_ACCURATE_RANGE = SCAN_RADIUS

sequence_num = args.sequence
print("sequence num: {}".format(sequence_num))

print("--- getting scan data -------------------------------------------------")

# get scan data for this sequence

poses_file_path = "./kitti_dso_files/results_files/seq_{}/poses_history_file.txt".format(sequence_num)
points_file_path = "./kitti_dso_files/results_files/seq_{}/pts_history_file.txt".format(sequence_num)

print("poses_file_path: {}".format(poses_file_path))
print("points_file_path: {}".format(points_file_path))

scan_dump_path = "./numpy_saves/{}_scans_100-45-45.pkl".format(sequence_num)
print("scan_dump_path: {}".format(scan_dump_path))

if (os.path.exists(scan_dump_path) == False) or args.force_regen_scan_data:
	print("regenerating scan data")
	# returns: frame_id, w2c_matrix, c2w_matrix, frame_pose, frame_points, scan_points
	out_dict = fetch_sequence_data(poses_file_path, points_file_path, NUM_SEARCH_FRAMES, SCAN_RADIUS, MAXIMUM_ACCURATE_RANGE, seq_num=sequence_num)
	pkl.dump(out_dict, open(scan_dump_path, "wb"))
else:
	out_dict = pkl.load(open(scan_dump_path, "rb"))
	print("loading existing scan data")

#out_dict = pkl.load(open("./numpy_saves/00_scans_100-45-45.old_data_prep.pkl", "rb"))

print("number of frames: {}".format(len(out_dict["frame_id"])))
print("first frame id: {} last frame id: {}".format(out_dict["frame_id"][0], out_dict["frame_id"][-1]))

'''
num_points = []
for arr in out_dict["scan_points"]:
	num_points.append(arr.shape[0])
print("points per scan: min {}, max {}, median {}".format(np.amin(num_points),np.amax(num_points),np.median(num_points)))
'''
print("--- getting ground truth kitti poses ----------------------------------")

# fetching ground truth kitti poses
print("fetching gt kitti pose data for seq {}".format(sequence_num))
gt_data_dict = fetch_gt_data("{}".format(sequence_num))

print("--- getting ground frame match data -----------------------------------")

# get ground truth match data (gt_match_dict)

gt_poses_file_path = "./kitti_dataset/dataset/poses/{}.txt".format(sequence_num)
print("gt_poses_file_path: {}".format(gt_poses_file_path))

gt_match_data_path = "./numpy_saves/{}_gt_10-100-false.pkl".format(sequence_num)
print("gt_match_data_path: {}".format(gt_match_data_path))

if (os.path.exists(gt_match_data_path) == False) or args.force_regen_match_data:
	print("regenerating ground truth match data")
	# returns: close_frames, close_frames_dists, strict_frames, strict_frames_dists
	gt_match_dict = get_groundtruth(gt_poses_file_path) # the indexes of this are the full set, no missing frames, so must be indexed by frame_id
	pkl.dump(gt_match_dict, open(gt_match_data_path, "wb"))
else:
	print("loading ground truth match data")
	gt_match_dict = pkl.load(open(gt_match_data_path, "rb"))

#---------------------------------------------------------------------------
print("--- starting engine ---------------------------------------------------")
#print("exiting since this is only a test\n\n\n")
#exit(0)

#if mode == "compare_match_pairs_clusters_grass_alignment":
if True:
	start = time.time()
	eng = matlab.engine.start_matlab()
	#eng = matlab.engine.start_matlab(stdout=io.StringIO())
	end = time.time()
	print("engine startup time: {}".format(end - start))
	
	#eng.cd(r'./matlab_scripts/', nargout=0)
	#addpath(r"my_folder")
	#addpath(genpath(r"my_folder_and_subfolders"))
	eng.addpath(r"./matlab_scripts/", nargout=0)
	eng.addpath(eng.genpath(r"./GrassGraphs/"))

	print("matlab search path:".format([x for x in eng.path().split(':') if "/usr/local" not in x]))
	for line in [x for x in eng.path().split(':') if "/usr/local" not in x]: print(line)
	print()

	grass_parameters = {
		'fhkt': 100.0,
		'Epsilon': 0.012,
		'hkt': 175.0,
		'GraphCase': 'epsilon',
		'GraphLapType': 'heatKer',
		'CorrMethod': 'minDistThrowOut',
		'CondFac': 500.0,
		'NumLBOEvec': 5.0,
		'EvecToMatch': matlab.double([[1.0,2.0,3.0]]),
		'DistScFac': 0.0001,
		'ScaleDist': 0.0,
		'ScType': 'maxScaling',
		'AllPerms': matlab.double([
			[1.0,1.0,1.0],
			[1.0,1.0,-1.0],
			[1.0,-1.0,1.0],
			[1.0,-1.0,-1.0],
			[-1.0,1.0,1.0],
			[-1.0,1.0,-1.0],
			[-1.0,-1.0,1.0],
			[-1.0,-1.0,-1.0]]),
		'NumPerms': 8.0,
		'eigSc': 0.0,
		'CorrThresh': 3.0,
		'ScoreType': 'NumCorr'
	}
'''
grass_shape_file_name = "COSEG.mat"
#grass_shape_file_name = "SHREC13.mat"
#eng.load('COSEG.mat')
#eng.evalc('load("{}")'.format(grass_shape_file_name))
#test_thing = eng.workspace['shapeCell']
test_thing1 = eng.importdata(grass_shape_file_name)
print(type(test_thing1))
print(np.asarray(test_thing1).shape)
print(np.asarray(test_thing1[0][0:5]))
'''

'''
eng.evalc('C = who;')
varnames = eng.workspace['C']
print(varnames)
'''

'''
coseg_set = np.asarray(eng.importdata("COSEG.mat"))
shrec13_set = np.asarray(eng.importdata("SHREC13.mat"))
combined_set = np.concatenate((coseg_set,shrec13_set), axis=0)

loaded_shapes = shrec13_set
max_radius_measurements = []
all_radius_measurements = np.asarray([])
for i in range(loaded_shapes.shape[0]):
	this_shape = loaded_shapes[i,:,:]

	# center shape on origin
	mean_coord = np.mean(this_shape, axis=0)
	this_shape = this_shape - mean_coord
	
	# find furthest point from center (largest radius)
	dists_from_center = np.linalg.norm(this_shape, axis=1)
	max_dist_from_center = np.max(dists_from_center)
	max_radius_measurements.append(max_dist_from_center)
	
	all_radius_measurements = np.append(all_radius_measurements,dists_from_center,axis=0)
	
	
	# visualize
	#pcd1 = o3d.geometry.PointCloud()
	#pcd1.points = o3d.utility.Vector3dVector(this_shape)
	#pcd1.paint_uniform_color([0.0,0.0,1.0])
	#coord1 = o3d.geometry.TriangleMesh().create_coordinate_frame(size=1.0)

	#o3d.visualization.draw_geometries([pcd1,coord1])

print("loaded_shapes.shape: {}".format(loaded_shapes.shape))

print("average maximum radius: {}".format(np.average(max_radius_measurements)))
print("maximum radius: {}".format(np.max(max_radius_measurements)))
plt.hist(max_radius_measurements)
plt.xlim(xmin=0.0, xmax = 20.0)

plt.figure()
plt.hist(all_radius_measurements)
plt.xlim(xmin=0.0, xmax = 20.0)
#plt.show()

# coseg average max radius: 14.781636437611361
# shrec13 average maximum radius: 14.862994866994185
# combined average maximum radius: 14.822315652302775
# maximum radius: 19.544303843660284

avg_dist = np.mean(all_radius_measurements)
std_dist = np.std(all_radius_measurements)

print("avg: {}".format(avg_dist))
print("avg +1: {}".format(avg_dist + 1*std_dist))
print("avg +2: {}".format(avg_dist + 2*std_dist))
print("avg +3: {}".format(avg_dist + 3*std_dist))

# avg: 8.293734108092474
# avg +1: 11.414523253162926
# avg +2: 14.535312398233376
# avg +3: 17.656101543303826

# for shrec13 set: 17.724776547614844

print("ding!")
assert(0)
'''
#---------------------------------------------------------------------------
# compare_match_pairs_alignment
# display the point cloud points of two scans, ground truth aligned

# converted*******************
# compare_match_pairs_clusters_alignment
# do clustering, display cluster centers of two scans, ground truth aligned

# compare_match_pairs_clusters_grass_alignment
# 

# ...

# converted*******************
# examine_match_pairs
# do clustering, display the points belonging to each cluster (not the cluster centers), two windows

# ...

# examine_clusters_points_pairs
# do clustering, display cluster points, two windows
#---------------------------------------------------------------------------

print("--- setting benchmark mode and generating frame lists -----------------")

# either want to look at a selection throughout the sequence, or pairs of matching ones

#TODO please label what each of these do/represent (and whether they're are up to date enough to work)

use_semantic_labels = args.use_semantic

if use_semantic_labels is None:
	# hardcode a which option to use by default here
	
	#use_semantic_labels = False #TODO
	use_semantic_labels = True
else:
	if use_semantic_labels == "yes":
		use_semantic_labels = True
	elif use_semantic_labels == "no":
		use_semantic_labels = False
	else:
		assert 0, "you used an incorrect input for specifying if semantic data should be used"

#print("use_semantic_labels: {}".format(use_semantic_labels))
#assert(0)

#TARGET_NUM = None # defaults to 120 in points_clustering_helper.py
#TARGET_NUM = 120
#TARGET_NUM = 20
if args.cluster_target_num is None:
	if use_semantic_labels == True:
		TARGET_NUM = 20
	else:
		TARGET_NUM = 120
else:
	TARGET_NUM = args.cluster_target_num

# hack added to turn total correction off in some very, very limited circumstances for tests/measurements that require it
# generally, turning total correction off will break grassgraph grassmannian association, and probably other stuff too (like other methods of association)
if args.use_uncorrected_totals is True: # defaults to False
	correct_num_clusters = False
else:
	correct_num_clusters = True # defaults to true

which_clustering_override = args.which_clustering_override

mode = args.operating_mode

if mode is None:
	#mode = "examine_clusters_sequence"
	#mode = "examine_match_pairs"									#TODO
	#mode = "examine_clusters_points_pairs"
	#mode = "compare_match_pairs_alignment"
	#mode = "compare_match_pairs_clusters_alignment"				#TODO
	#mode = "compare_match_pairs_clusters_grass_alignment"
	#mode = "compare_match_pairs_clusters_o3d_alignment"
	#mode = "compare_match_pairs_clusters_clipper_alignment" #****************** TODO: try tuning this?

	mode = "compare_match_pairs_clusters_alignment_measurement"
	#mode = "test_real_and_synthetic_landmark_association"
	#mode = "test_real_and_synthetic_landmark_association_grassgraph_data"
	#mode = "test_real_and_synthetic_landmark_association_display"
	
print("mode is {}".format(mode))

# list_of_scan_indexes is indexing into the list of keyframes returned by DSO. they are not kitti frame id numbers

	
if mode == "examine_clusters_sequence":
	# this is a collection of indexes into the sequence from DSO, to check that clustering works for keyframes (and their pseudo-scans) throughout the sequence
	list_of_scan_indexes = list(range(100, 1100, 100)) #list(range(100, 110, 1))
else:
	if ((mode == "examine_match_pairs") or \
		(mode == "examine_clusters_points_pairs") or \
		(mode == "compare_match_pairs_alignment") or \
		(mode == "compare_match_pairs_clusters_alignment") or \
		(mode == "compare_match_pairs_clusters_grass_alignment") or \
		(mode == "compare_match_pairs_clusters_o3d_alignment") or \
		(mode == "compare_match_pairs_clusters_clipper_alignment")) and not args.force_fullseq_pair_gen:
		
		print("doing old pair generation *******")
		
		# 15 equally spaced frames during the longest place recognition sequence in kitti seq00
		# these values are indexed into the array of keyframes we get from DSO, which does not contain every frame in the sequence
		list_of_scan_indexes = list(range(2500,2940,36)) #list(range(2500,3040,36)) #TODO what is the significance of this range? a long section in seq00 that has revisits?
		
		# get ground truth match data
		# returns: close_frames, close_frames_dists, strict_frames, strict_frames_dists
		#gt_match_dict = get_groundtruth(gt_poses_file_path) # the indexes of this are the full set, no missing frames, so must be indexed by frame_id
		#pkl.dump(gt_match_dict, open("./numpy_saves/00_gt_10-100-false.pkl", "wb"))
		#gt_match_dict = pkl.load(open("./numpy_saves/00_gt_10-100-false.pkl", "rb"))
		
		# gt_match_dict was obtained for the current particular sequence above
		
		'''
		#for i in range(len(gt_match_dict["strict_frames"])):
		#	print(gt_match_dict["strict_frames"][i])
		print((np.asarray(gt_match_dict["strict_frames"]) > 10).sum())
		print(len(gt_match_dict["strict_frames"]))
		assert(0)
		'''
		
		# for those indexes, we need to get their best-matching frame(s) and find the closest DSO indexes that exist
		# this allows us to examine how consistent the clustering is between two visits that are completely unrelated in time
		list_of_scan_indexes2 = []
		for i in range(len(list_of_scan_indexes)):
			dso_index = list_of_scan_indexes[i]
			frame_id = out_dict["frame_id"][dso_index]
			
			# fetch the ground truth frame that's closest to this DSO-provided keyframe, by the keyframe's kitti frame id
			# the indexes of gt_match_dict are the full set, no missing frames, so must be indexed by frame_id
			closest_gt_match = gt_match_dict["strict_frames"][frame_id]
			
			# find the DSO keyframe that's closest to that, out of the frames chosen by DSO as keyframes
			dso_frame_ids = out_dict["frame_id"]
			distances = np.abs(np.asarray(dso_frame_ids) - closest_gt_match)
			closest_dso_index = np.argmin(distances)
			
			list_of_scan_indexes2.append(closest_dso_index)
			#print("for DSO index {} with frame_id *{}*, the closest gt match is frame_id *{}*, and the closest DSO frame to that is {} with id *{}*".format(
			#	dso_index,frame_id,closest_gt_match,closest_dso_index,out_dict["frame_id"][closest_dso_index]))
			
			# if you get "for DSO index 2968 with frame_id 3862, the closest gt match is frame_id -1, and the closest DSO frame to that is 0 with id 0"
			# then that means there is no gt match for this frame in the DSO sequence (it's never revisited)
	elif mode == "test_real_and_synthetic_landmark_association_grassgraph_data":
		#TODO special case where our data comes provided by the authors of grassgraph
		
		coseg_set = np.asarray(eng.importdata("COSEG.mat"))
		shrec13_set = np.asarray(eng.importdata("SHREC13.mat"))
		combined_set = np.concatenate((coseg_set,shrec13_set), axis=0)
		
		#TODO use this to set what set to use
		#loaded_shapes = coseg_set
		loaded_shapes = shrec13_set
		#loaded_shapes = combined_set
		
		#TODO this is where you can change the amount of sampling, as there are by default more than 120 points
		rng = np.random.default_rng(5)
		need_points_per_shape = 120
		
		num_points_per_shape = loaded_shapes.shape[1]
		num_points_remove = num_points_per_shape-need_points_per_shape
		decimated_shapes = []
		for i in range(loaded_shapes.shape[0]):
			temp_shape = loaded_shapes[i,:,:]
			del_sel = np.sort(rng.choice(temp_shape.shape[0],size=num_points_remove,replace=False))
			temp_shape = np.delete(temp_shape, del_sel, axis=0)
			decimated_shapes.append(temp_shape)
		loaded_shapes = np.asarray(decimated_shapes) #TODO ************************************************************************************************************* This was the difference between the subsampled 120 and not
																																										# subscampled SHREC runs (250 points vs 120 with kitti)
		
		list_of_scan_indexes = list(range(loaded_shapes.shape[0]))
		
		shapes_loaded_from_an_alternate_source = loaded_shapes
		
		#print("haven't implemented fetching of grassgraph data yet!!!")
		#assert(0)
		
		
	#elif mode == "compare_match_pairs_clusters_alignment_measurement":
	#elif mode == "test_real_and_synthetic_landmark_association":
	#elif mode == "test_real_and_synthetic_landmark_association_display":
	else:
		# get equally space frames throughout the sequence
		
		# find the ones that have a (closest) gt match
		
		# record those and the gt match in list_of_scan_indexes and list_of_scan_indexes2
		
		#-----------------------
		print("doing new pair generation *******")
		
		# get every nth index throughout the DSO keyframe sequence
		skip_every_n = args.skip_num_frames #10
		temp_list_of_scan_indexes = list(range(0,len(out_dict["frame_id"]),skip_every_n))
		#print(temp_list_of_scan_indexes)
		#assert(0)
		
		list_of_scan_indexes = []
		list_of_scan_indexes2 = []
		#check if it has a closest match in the gt match data
		for i in range(len(temp_list_of_scan_indexes)):
			dso_index = temp_list_of_scan_indexes[i]
			frame_id = out_dict["frame_id"][dso_index]
			
			#print("dso index: {}".format(dso_index))
			#print("frame_id: {}".format(frame_id))
			
			# fetch the ground truth frame that's closest to this DSO-provided keyframe, by the keyframe's kitti frame_id
			# the indexes of gt_match_dict are the full set, no missing frames, so must be indexed by frame_id
			# the returned index is -1 if there's no nearby frame, otherwise it's an index into kitti sequence frames (of which there are more than DSO keyframes)
			closest_gt_match = gt_match_dict["strict_frames"][frame_id]
			
			#print("closest_gt_match: {}".format(closest_gt_match))
			
			if closest_gt_match != -1:
				#if there is in fact a closest gt match to this frame
				
				# there are fewer DSO keyframes than kitti frames, so we need to find the dso keyframe with the closest frame_id to the kitti frame
				dso_frame_ids = out_dict["frame_id"]
				distances = np.abs(np.asarray(dso_frame_ids) - closest_gt_match)
				closest_dso_index = np.argmin(distances)
				
				#print("closest_dso_index: {}".format(closest_dso_index))
				#print()
				
				list_of_scan_indexes.append(dso_index)
				list_of_scan_indexes2.append(closest_dso_index)
			#else:
			#	dso_index = temp_list_of_scan_indexes[i]
			#	frame_id = out_dict["frame_id"][dso_index]
			#	print("index {} (id {}) has no matching revisit".format(dso_index,frame_id))
		
		
		#print(list_of_scan_indexes)
		#print("")
		#print(list_of_scan_indexes2)
		#print("done special case for measurement")
		#assert(0)
		print("number of frames with revisits gathered: {}".format(len(list_of_scan_indexes)))

#assert(0)

'''
# print out the lists of frames to be compared
print("list_of_scan_indexes (dso indexes): {}".format(list_of_scan_indexes))
print("list_of_scan_indexes2 (dso indexes): {}".format(list_of_scan_indexes2))
list1 = []
list2 = []
for i in range(len(list_of_scan_indexes)):
	list1.append(out_dict["frame_id"][list_of_scan_indexes[i]])
	list2.append(out_dict["frame_id"][list_of_scan_indexes2[i]])
print()
print("list_of_scan_indexes (kitti indexes): {}".format(list1))
print("list_of_scan_indexes2 (kitti indexes): {}".format(list2))
print()
#assert(0)
'''

#---------------------------------------------------------------------------
'''
# returns: numeric label for each point in scan_points. -1 means unclustered (outlier or rejected)
def cluster_points(scan_points, frame_xyz):
	
	#DBSCAN(eps=1, min_samples=5)
	#DBSCAN(eps=0.5, min_samples=5)
	
	#clustering = DBSCAN(eps=0.75, min_samples=5).fit(scan_points)
	clustering = Birch(threshold=0.5, n_clusters=120).fit(scan_points)
	labels = clustering.labels_
	points_out = scan_points
	
	#pcd = o3d.geometry.PointCloud()
	#pcd.points = o3d.utility.Vector3dVector(scan_points)
	#points_out = np.asarray(pcd.voxel_down_sample(voxel_size=3).points)
	##points_out = np.asarray(pcd.voxel_down_sample(voxel_size=10).points)
	#labels = np.ones((points_out.shape[0]),dtype="int")
	#print("number of points in downsampled point cloud: {}".format(points_out.shape[0]))
	
	return labels,points_out
'''
#---------------------------------------------------------------------------
'''
colors_list = np.asarray([[0.8500, 0.3250, 0.0980],
						[0.9290, 0.6940, 0.1250],
						[0.4940, 0.1840, 0.5560],
						[0.4660, 0.6740, 0.1880],
						[0.3010, 0.7450, 0.9330],
						[0.6350, 0.0780, 0.1840]])
'''
colors_list = np.asarray([[47,79,79],[46,139,87],[25,25,112],[139,0,0],
							[128,128,0],[255,0,0],[255,140,0],[0,255,0],
							[186,85,211],[0,250,154],[0,255,255],[0,0,255],
							[240,128,128],[255,0,255],[30,144,255],[255,255,84],
							[221,160,221],[255,20,147],[245,222,179],[135,206,250]])/255.0

print("-----------------------------------------------------------------------")
print("--- running benchmark -------------------------------------------------")
print("-----------------------------------------------------------------------")
#print("exiting since this is only a test\n\n\n")
#exit(0)

if (mode == "examine_match_pairs") or (mode == "examine_clusters_points_pairs"):
	vis = o3d.visualization.Visualizer()
	vis2 = o3d.visualization.Visualizer()
	vis.create_window(window_name='TopLeft', width=960, height=540, left=0, top=0)
	vis2.create_window(window_name='TopRight', width=960, height=540, left=960, top=0)
	print("ctrl+c for next")
	print("ctrl+\ to exit")


# record for each pair of scans: 
#	number of points (?)
#	number of clusters
#	number of outliers (range/neighbors),
#	distances between close ones (based on range/neighbors)
results_recording_dict = {
	"frame_1_max":[],			# the maximum limits of points in the frame
	"frame_1_min":[],
	"frame_2_max":[],
	"frame_2_min":[],
	"num_points_1":[],			# the number of points in the frame. together with the bounds of the bounding box above, rough density can be computed
	"num_points_2":[],
	
	"num_clusters" : [],		# should be the same for both frames, technically for all frames
	"num_outliers_nn" : [],		# number of cluster points not in a mutual-nn relationship. should be same between sets, see below
	"dists_nearest_neighbors" : [],	# for those in a nn relationship, how far apart are they?
	
	"dists_to_closest" : []		# record all the distance to the nearest point for all points in frame 1, we can apply range gating later, or just plot a distribution
								# we only record the number of outliers that don't have a pairing within 1m in frame 1, we'll get to the reverse pairing where we consider frame 2 later in the sequence
	
#	"num_outliers_range_1" : [],	# we only record the number of outliers that don't have a pairing within 1m in frame 1, we'll get to the reverse pairing where we consider frame 2 later in the sequence
#	"num_outliers_range_2" : [],
#	"dists_range_gate" : []			# for those with a cluster from the other set within the range gate, how far apart are they?
}

# mutual nearest neighbor are mutual, so it's 1-1 between set 1 and set 2
# for "circles" and "triangles": any time a circle is in a nn relationship, it's taking with it exactly one triangle and vise-versa
# because the number of circles and triangles are equal, that means the same number of remainders must be present in both sets

# used for evaluation of assoc method performance
results_recording_dict_2 = {
	"num_matches_list":					[],
	"inlier_frob_norm_list":			[],
	"fullset_frob_norm_list":			[],
	"num_matches_total_possible_list":	[],
	"matrix_frob_list":					[],
	"rotation_difference_angle_list":	[]
}
bad_matrix_count = 0
matrix_count = 0

times_taken = []
# cluster scans
#for i in range(len(out_dict["frame_id"])):
temp_loop_idx = -1 #TODO used in dumb hackery for display and image generation purposes
for i in range(len(list_of_scan_indexes)):
	temp_loop_idx += 1
	if mode == "compare_match_pairs_alignment":
		dso_frame_num1 = list_of_scan_indexes[i]				# list of scan indexes is not nesisarily sequential. "i" is the position in this list, not the specific index of a keyframe from DSO
		frame_id1 = out_dict["frame_id"][dso_frame_num1]
		scan_points1 = out_dict["scan_points"][dso_frame_num1]
		#frame_xyz1 = out_dict["frame_pose"][dso_frame_num1]		# according to DSO
		w2c_matrix1 = out_dict["w2c_matrix"][dso_frame_num1]
		c2w_kitti1 = gt_data_dict["kitti_c2w"][frame_id1]
		
		dso_frame_num2 = list_of_scan_indexes2[i]
		frame_id2 = out_dict["frame_id"][dso_frame_num2]
		scan_points2 = out_dict["scan_points"][dso_frame_num2]
		#frame_xyz2 = out_dict["frame_pose"][dso_frame_num2]		# according to DSO
		w2c_matrix2 = out_dict["w2c_matrix"][dso_frame_num2]
		c2w_kitti2 = gt_data_dict["kitti_c2w"][frame_id2]
		
		#would_like_to_show_semantic_colors = True
		would_like_to_show_semantic_colors = False
		
		#show_only_scan_1 = True
		show_only_scan_1 = False
		
		if would_like_to_show_semantic_colors:
			scan_labels1 = out_dict["scan_labels"][dso_frame_num1]
			scan_labels2 = out_dict["scan_labels"][dso_frame_num2]
			#print("scan_labels1 {} scan_labels2 {}".format(len(scan_labels1),len(scan_labels2)))
			#print("scan_points1 {} scan_points2 {}".format(len(scan_points1),len(scan_points2)))
			#print(scipy.stats.mode(scan_labels1)[0])
			#print(scipy.stats.mode(scan_labels2)[0])
			#print(scan_labels1.tolist())
			#print()
			point_colors1 = colors_list[((scan_labels1+1)%colors_list.shape[0]),:]
			point_colors1[scan_labels1 == -1] = [0,0,0]
			point_colors2 = colors_list[((scan_labels2+1)%colors_list.shape[0]),:]
			point_colors2[scan_labels2 == -1] = [0,0,0]
		
		
		scanpts1_h = np.concatenate((scan_points1, np.ones((scan_points1.shape[0],1))), axis=1)
		scanpts1_cam_h = np.matmul(w2c_matrix1,np.transpose(scanpts1_h))
		scanpts1_global_h = np.matmul(c2w_kitti1,scanpts1_cam_h)
		scanpts1_global = np.transpose(scanpts1_global_h)[:,0:3]
		
		scanpts2_h = np.concatenate((scan_points2, np.ones((scan_points2.shape[0],1))), axis=1)
		scanpts2_cam_h = np.matmul(w2c_matrix2,np.transpose(scanpts2_h))
		scanpts2_global_h = np.matmul(c2w_kitti2,scanpts2_cam_h)
		scanpts2_global = np.transpose(scanpts2_global_h)[:,0:3]
		
		pcd1 = o3d.geometry.PointCloud()
		pcd1.points = o3d.utility.Vector3dVector(scanpts1_global)
		if would_like_to_show_semantic_colors:
			pcd1.colors = o3d.utility.Vector3dVector(point_colors1)
		else:
			pcd1.paint_uniform_color([1.0,0.0,0.0])
		coord1 = o3d.geometry.TriangleMesh().create_coordinate_frame(size=10.0,origin=gt_data_dict["kitti_xyz"][frame_id1])
		
		if not show_only_scan_1:
			pcd2 = o3d.geometry.PointCloud()
			pcd2.points = o3d.utility.Vector3dVector(scanpts2_global)
			if would_like_to_show_semantic_colors:
				pcd2.colors = o3d.utility.Vector3dVector(point_colors2)
			else:
				pcd2.paint_uniform_color([0.0,0.0,1.0])
			coord2 = o3d.geometry.TriangleMesh().create_coordinate_frame(size=10.0,origin=gt_data_dict["kitti_xyz"][frame_id2])
		
		if not show_only_scan_1:
			o3d.visualization.draw_geometries([pcd1,pcd2,coord1,coord2])
		else:
			o3d.visualization.draw_geometries([pcd1,coord1])
		
	elif mode == "compare_match_pairs_clusters_alignment":
		dso_frame_num1 = list_of_scan_indexes[i]				# list of scan indexes is not nesisarily sequential. "i" is the position in this list, not the specific index of a keyframe from DSO
		frame_id1 = out_dict["frame_id"][dso_frame_num1]
		scan_points1 = out_dict["scan_points"][dso_frame_num1]
		#frame_xyz1 = out_dict["frame_pose"][dso_frame_num1]		# according to DSO
		w2c_matrix1 = out_dict["w2c_matrix"][dso_frame_num1]
		c2w_kitti1 = gt_data_dict["kitti_c2w"][frame_id1]
		
		dso_frame_num2 = list_of_scan_indexes2[i]
		frame_id2 = out_dict["frame_id"][dso_frame_num2]
		scan_points2 = out_dict["scan_points"][dso_frame_num2]
		#frame_xyz2 = out_dict["frame_pose"][dso_frame_num2]		# according to DSO
		w2c_matrix2 = out_dict["w2c_matrix"][dso_frame_num2]
		c2w_kitti2 = gt_data_dict["kitti_c2w"][frame_id2]
		
		method_string = "cluster"
		#method_string = "pcsimp"
		
		#use_semantic_labels = False
		
		if use_semantic_labels:
			scan_labels1 = out_dict["scan_labels"][dso_frame_num1]
			scan_labels2 = out_dict["scan_labels"][dso_frame_num2]
		
		# get clusters
		if use_semantic_labels:
			cluster_dict = do_clustering({"scan_points": scan_points1,"scan_labels":scan_labels1,"target_num":TARGET_NUM, "which_clustering_override":which_clustering_override}, method=method_string, remove_noise=True, eng=eng)
		else:
			cluster_dict = do_clustering({"scan_points": scan_points1,"target_num":TARGET_NUM, "which_clustering_override":which_clustering_override}, method=method_string, remove_noise=True, eng=eng)
		cluster_points1 = cluster_dict["cluster_points"]
		if use_semantic_labels:
			cluster_dict = do_clustering({"scan_points": scan_points2,"scan_labels":scan_labels2,"target_num":TARGET_NUM, "which_clustering_override":which_clustering_override}, method=method_string, remove_noise=True, eng=eng)
		else:
			cluster_dict = do_clustering({"scan_points": scan_points2,"target_num":TARGET_NUM, "which_clustering_override":which_clustering_override}, method=method_string, remove_noise=True, eng=eng)
		cluster_points2 = cluster_dict["cluster_points"]
		
		# transfer from keyframe coords to global coord using gt (simulating grass alignment)
		scanpts1_h = np.concatenate((cluster_points1, np.ones((cluster_points1.shape[0],1))), axis=1)
		scanpts1_cam_h = np.matmul(w2c_matrix1,np.transpose(scanpts1_h))
		scanpts1_global_h = np.matmul(c2w_kitti1,scanpts1_cam_h)
		scanpts1_global = np.transpose(scanpts1_global_h)[:,0:3]
		
		scanpts2_h = np.concatenate((cluster_points2, np.ones((cluster_points2.shape[0],1))), axis=1)
		scanpts2_cam_h = np.matmul(w2c_matrix2,np.transpose(scanpts2_h))
		scanpts2_global_h = np.matmul(c2w_kitti2,scanpts2_cam_h)
		scanpts2_global = np.transpose(scanpts2_global_h)[:,0:3]
		
		# find mutual correspondances
		# nearest 2 to each 1
		dist_2near1, ind_2near1 = neighbors.NearestNeighbors().fit(scanpts2_global).kneighbors(scanpts1_global, n_neighbors=1, return_distance=True)
		ind_2near1 = ind_2near1[:,0]
		#print(ind_2near1)
		
		# nearest 1 to each 2
		dist_1near2, ind_1near2 = neighbors.NearestNeighbors().fit(scanpts1_global).kneighbors(scanpts2_global, n_neighbors=1, return_distance=True)
		ind_1near2 = ind_1near2[:,0]
		#print(ind_1near2)

		mut_corr = []
		for i in range(len(ind_2near1)):
			point1_idx = i
			close2_idx = ind_2near1[i]
			close1_idx = ind_1near2[close2_idx]
			if close1_idx == i:
				mut_corr.append([point1_idx, close2_idx])
		mut_corr = np.asarray(mut_corr)
		print("number of mutual correspondances: {}".format(mut_corr.shape[0]))
		
		both_points = np.concatenate((scanpts1_global,scanpts2_global), axis=0)
		lines_idx = mut_corr.copy()
		lines_idx[:,1] += scanpts1_global.shape[0] # points2 was concatenated to the end, past points1
		corr_line_set = o3d.geometry.LineSet(
			points=o3d.utility.Vector3dVector(both_points),
			lines=o3d.utility.Vector2iVector(lines_idx),
		)
		
		# visualize
		pcd1 = o3d.geometry.PointCloud()
		pcd1.points = o3d.utility.Vector3dVector(scanpts1_global)
		pcd1.paint_uniform_color([1.0,0.0,0.0])
		coord1 = o3d.geometry.TriangleMesh().create_coordinate_frame(size=10.0,origin=gt_data_dict["kitti_xyz"][frame_id1])
		
		pcd2 = o3d.geometry.PointCloud()
		pcd2.points = o3d.utility.Vector3dVector(scanpts2_global)
		pcd2.paint_uniform_color([0.0,0.0,1.0])
		coord2 = o3d.geometry.TriangleMesh().create_coordinate_frame(size=10.0,origin=gt_data_dict["kitti_xyz"][frame_id2])
		
		o3d.visualization.draw_geometries([pcd1,pcd2,coord1,coord2,corr_line_set])
	
	elif mode == "compare_match_pairs_clusters_grass_alignment":
		dso_frame_num1 = list_of_scan_indexes[i]				# list of scan indexes is not nesisarily sequential. "i" is the position in this list, not the specific index of a keyframe from DSO
		frame_id1 = out_dict["frame_id"][dso_frame_num1]
		scan_points1 = out_dict["scan_points"][dso_frame_num1]
		#frame_xyz1 = out_dict["frame_pose"][dso_frame_num1]		# according to DSO
		#w2c_matrix1 = out_dict["w2c_matrix"][dso_frame_num1]
		#c2w_kitti1 = gt_data_dict["kitti_c2w"][frame_id1]
		
		dso_frame_num2 = list_of_scan_indexes2[i]
		frame_id2 = out_dict["frame_id"][dso_frame_num2]
		scan_points2 = out_dict["scan_points"][dso_frame_num2]
		#frame_xyz2 = out_dict["frame_pose"][dso_frame_num2]		# according to DSO
		#w2c_matrix2 = out_dict["w2c_matrix"][dso_frame_num2]
		#c2w_kitti2 = gt_data_dict["kitti_c2w"][frame_id2]
		
		method_string = "cluster"
		#method_string = "pcsimp"
		
		if use_semantic_labels:
			scan_labels1 = out_dict["scan_labels"][dso_frame_num1]
			scan_labels2 = out_dict["scan_labels"][dso_frame_num2]
		
		# get clusters
		if use_semantic_labels:
			cluster_dict = do_clustering({"scan_points": scan_points1,"scan_labels":scan_labels1,"target_num":TARGET_NUM, "which_clustering_override":which_clustering_override}, method=method_string, remove_noise=True, eng=eng)
		else:
			cluster_dict = do_clustering({"scan_points": scan_points1,"target_num":TARGET_NUM, "which_clustering_override":which_clustering_override}, method=method_string, remove_noise=True, eng=eng)
		cluster_points1 = cluster_dict["cluster_points"]
		if use_semantic_labels:
			cluster_dict = do_clustering({"scan_points": scan_points2,"scan_labels":scan_labels2,"target_num":TARGET_NUM, "which_clustering_override":which_clustering_override}, method=method_string, remove_noise=True, eng=eng)
		else:
			cluster_dict = do_clustering({"scan_points": scan_points2,"target_num":TARGET_NUM, "which_clustering_override":which_clustering_override}, method=method_string, remove_noise=True, eng=eng)
		cluster_points2 = cluster_dict["cluster_points"]
		
		# get clusters
		#cluster_dict = do_clustering({"scan_points": scan_points1,"target_num":TARGET_NUM}, method="cluster", remove_noise=True)
		#cluster_points1 = cluster_dict["cluster_points"]
		#cluster_dict = do_clustering({"scan_points": scan_points2,"target_num":TARGET_NUM}, method="cluster", remove_noise=True)
		#cluster_points2 = cluster_dict["cluster_points"]
		
		# trying in case shifting to origin makes a difference
		# it appears it does not, as to be expected for an inter-point-distance based method which computes representations independantly
		#average_1 = np.average(cluster_points1, axis=0)
		#average_2 = np.average(cluster_points2, axis=0)
		#cluster_points1 = cluster_points1 - average_1
		#cluster_points2 = cluster_points2 - average_2
		
		# testing with test pointclouds
		'''
		X = np.asarray(eng.dump_test_shapes({"shape_num": 40})["shape"])
		Y = np.asarray(eng.dump_test_shapes({"shape_num": 40})["shape"])
		
		A = np.asarray(eng.affineTransformation3D_Clean(
			matlab.double([0, 0, np.pi/8]),			#thetas
			matlab.double([0, -np.pi/8, np.pi/6]),	#sigmas
			matlab.double([0, 0, 0]),				#translation
			matlab.double([2, 0.5, -1])				#scaling
		))
		
		Yh = np.concatenate((Y, np.ones((Y.shape[0],1))), axis=1)
		Y_aff = np.matmul(Yh,A)
		Y_aff = Y_aff[:,0:3]
		
		rng = np.random.default_rng()
		rng.shuffle(Y_aff)				# in-place shuffle of Y_aff
		Y_aff = np.ascontiguousarray(Y_aff)
		cluster_points1 = X
		cluster_points2 = Y_aff
		'''
		
		# try removing offset before matching
		#cluster_points1 = cluster_points1 - (np.amin(cluster_points1, axis=0)+np.amax(cluster_points1, axis=0))/2.0
		#cluster_points2 = cluster_points2 - (np.amin(cluster_points2, axis=0)+np.amax(cluster_points2, axis=0))/2.0
		
		# attempt grassmannian alignment
		#start = time.time()
		out_dict2 = eng.grassgraph_assoc_test({"X":matlab.double(cluster_points1),"Y":matlab.double(cluster_points2),"p":grass_parameters})
		#end = time.time()
		#print("engine function time: {}".format(end - start))
		rA = np.asarray(out_dict2["rA"])
		print("rA matrix: {}".format(rA))
		print("num_matches: {}".format(out_dict2["num_matches"]))
		
		# Y = X*rA
		# pts2 = pts1*rA
		
		cluster_points1_h = np.concatenate((cluster_points1,np.ones((cluster_points1.shape[0],1))), axis=1)
		cluster_points1_h = np.matmul(cluster_points1_h,rA)
		cluster_points1 = cluster_points1_h[:,0:3] #+0.5
		
		# find mutual correspondances
		# nearest 2 to each 1
		dist_2near1, ind_2near1 = neighbors.NearestNeighbors().fit(cluster_points2).kneighbors(cluster_points1, n_neighbors=1, return_distance=True)
		ind_2near1 = ind_2near1[:,0]
		#print(ind_2near1)
		
		# nearest 1 to each 2
		dist_1near2, ind_1near2 = neighbors.NearestNeighbors().fit(cluster_points1).kneighbors(cluster_points2, n_neighbors=1, return_distance=True)
		ind_1near2 = ind_1near2[:,0]
		#print(ind_1near2)

		mut_corr = []
		for i in range(len(ind_2near1)):
			point1_idx = i
			close2_idx = ind_2near1[i]
			close1_idx = ind_1near2[close2_idx]
			if close1_idx == i:
				mut_corr.append([point1_idx, close2_idx])
		mut_corr = np.asarray(mut_corr)
		print("number of mutual correspondances: {}".format(mut_corr.shape[0]))
		
		both_points = np.concatenate((cluster_points1,cluster_points2), axis=0)
		lines_idx = mut_corr.copy()
		lines_idx[:,1] += cluster_points1.shape[0] # points2 was concatenated to the end, past points1
		corr_line_set = o3d.geometry.LineSet(
			points=o3d.utility.Vector3dVector(both_points),
			lines=o3d.utility.Vector2iVector(lines_idx),
		)
		
		# visualize
		pcd1 = o3d.geometry.PointCloud()
		pcd1.points = o3d.utility.Vector3dVector(cluster_points1)
		pcd1.paint_uniform_color([1.0,0.0,0.0])
		#coord1 = o3d.geometry.TriangleMesh().create_coordinate_frame(size=10.0,origin=gt_data_dict["kitti_xyz"][frame_id1])
		
		pcd2 = o3d.geometry.PointCloud()
		pcd2.points = o3d.utility.Vector3dVector(cluster_points2)
		pcd2.paint_uniform_color([0.0,0.0,1.0])
		#coord2 = o3d.geometry.TriangleMesh().create_coordinate_frame(size=10.0,origin=gt_data_dict["kitti_xyz"][frame_id2])
		
		o3d.visualization.draw_geometries([pcd1,pcd2,corr_line_set])

	elif mode == "compare_match_pairs_clusters_o3d_alignment":
		dso_frame_num1 = list_of_scan_indexes[i]				# list of scan indexes is not nesisarily sequential. "i" is the position in this list, not the specific index of a keyframe from DSO
		frame_id1 = out_dict["frame_id"][dso_frame_num1]
		scan_points1 = out_dict["scan_points"][dso_frame_num1]
		#frame_xyz1 = out_dict["frame_pose"][dso_frame_num1]		# according to DSO
		#w2c_matrix1 = out_dict["w2c_matrix"][dso_frame_num1]
		#c2w_kitti1 = gt_data_dict["kitti_c2w"][frame_id1]
		
		dso_frame_num2 = list_of_scan_indexes2[i]
		frame_id2 = out_dict["frame_id"][dso_frame_num2]
		scan_points2 = out_dict["scan_points"][dso_frame_num2]
		#frame_xyz2 = out_dict["frame_pose"][dso_frame_num2]		# according to DSO
		#w2c_matrix2 = out_dict["w2c_matrix"][dso_frame_num2]
		#c2w_kitti2 = gt_data_dict["kitti_c2w"][frame_id2]
		
		# get clusters
		cluster_dict = do_clustering({"scan_points": scan_points1,"target_num":TARGET_NUM}, method="cluster", remove_noise=True)
		cluster_points1 = cluster_dict["cluster_points"]
		cluster_dict = do_clustering({"scan_points": scan_points2,"target_num":TARGET_NUM}, method="cluster", remove_noise=True)
		cluster_points2 = cluster_dict["cluster_points"]
		
		# try removing offset before matching
		scan_points1 = scan_points1 - (np.amin(scan_points1, axis=0)+np.amax(scan_points1, axis=0))/2.0
		scan_points2 = scan_points2 - (np.amin(scan_points2, axis=0)+np.amax(scan_points2, axis=0))/2.0
		
		# load pointclouds into o3d
		frame1_pcd = o3d.geometry.PointCloud()
		frame1_pcd.points = o3d.utility.Vector3dVector(scan_points1)
		frame2_pcd = o3d.geometry.PointCloud()
		frame2_pcd.points = o3d.utility.Vector3dVector(scan_points2)
		
		# attempt o3d global alignment (and maybe ICP)
		#voxel_size = 0.05  # means 5cm for this dataset
		#radius_normal = voxel_size * 2
		#radius_feature = voxel_size * 5
		#radius_normal = 0.33
		#radius_feature = 1.0

		# 1/2 to 2/3 were pretty good
		#radius_normal = 1.0
		#radius_feature = 3.0
	
		#voxel_size = 0.1
		#radius_normal = 1.5
		#radius_feature = 4.0
		
		# this combination seems to work fairly well, with an iteration time of 0.064
		#10,000 iterations (10x reduction from default)
		voxel_size = 0.75
		radius_normal = 1.5
		radius_feature = 5.0
		
		# takes longer ~0.19 and aligns more poorly
		# trying to match un-like pointclouds also takes around 0.2 seconds
		# in other words, it only takes 0.06 seconds (as above) when things align well, which is the exception rather than the rule
		#voxel_size = 0.1
		#radius_normal = 1.5
		#radius_feature = 5.0
		
		# de-noise pointclouds
		temp_cloud, ind = frame1_pcd.remove_radius_outlier(nb_points=5, radius=2.5)
		frame1_pcd = temp_cloud
		temp_cloud, ind = frame2_pcd.remove_radius_outlier(nb_points=5, radius=2.5)
		frame2_pcd = temp_cloud
		
		# prep pointclouds for global registration
		frame1_pcd_down = frame1_pcd.voxel_down_sample(voxel_size)
		temp_cloud, ind = frame1_pcd_down.remove_radius_outlier(nb_points=5, radius=2.5)
		frame1_pcd_down = temp_cloud
		frame1_pcd_down.estimate_normals(
			o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
		)
		frame1_pcd_down_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
			frame1_pcd_down,
			o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
		)
		
		frame2_pcd_down = frame2_pcd.voxel_down_sample(voxel_size)
		temp_cloud, ind = frame2_pcd_down.remove_radius_outlier(nb_points=5, radius=2.5)
		frame2_pcd_down = temp_cloud
		frame2_pcd_down.estimate_normals(
			o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
		)
		frame2_pcd_down_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
			frame2_pcd_down,
			o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
		)
		
		# RANSAC global alignment
		def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
			distance_threshold = voxel_size * 1.5
			print(":: RANSAC registration on downsampled point clouds.")
			print("   Since the downsampling voxel size is %.3f," % voxel_size)
			print("   we use a liberal distance threshold %.3f." % distance_threshold)
			result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
				source_down,
				target_down,
				source_fpfh,
				target_fpfh,
				True,
				distance_threshold,
				o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
				3,
				[
				    o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
				    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
				],
				#o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
				o3d.pipelines.registration.RANSACConvergenceCriteria(10000, 0.999)
			)
			return result
		
		# faster global alignment
		def execute_fast_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
			distance_threshold = voxel_size * 0.5
			print(":: Apply fast global registration with distance threshold %.3f" % distance_threshold)
			result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
				source_down,
				target_down,
				source_fpfh,
				target_fpfh,
				o3d.pipelines.registration.FastGlobalRegistrationOption(maximum_correspondence_distance=distance_threshold))
			return result
		
		# ICP improvement
		def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size, result_global):
			distance_threshold = voxel_size * 0.4
			print(":: Point-to-plane ICP registration is applied on original point")
			print("   clouds to refine the alignment. This time we use a strict")
			print("   distance threshold %.3f." % distance_threshold)
			result = o3d.pipelines.registration.registration_icp(
				source,
				target,
				distance_threshold,
				result_global.transformation,
				#o3d.pipelines.registration.TransformationEstimationPointToPlane()
				o3d.pipelines.registration.TransformationEstimationPointToPoint()
			)
			return result
		
		# do global registration
		start = time.time()
		register_result = execute_global_registration(frame1_pcd_down, frame2_pcd_down, frame1_pcd_down_fpfh, frame2_pcd_down_fpfh, voxel_size)
		#register_result = execute_fast_global_registration(frame1_pcd_down, frame2_pcd_down, frame1_pcd_down_fpfh, frame2_pcd_down_fpfh, voxel_size)
		print("Global registration took %.3f sec.\n" % (time.time() - start))
		times_taken.append(time.time() - start)
		
		# try refining with ICP
		'''
		frame1_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
		frame2_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
		register_result = refine_registration(frame1_pcd, frame2_pcd, None, None, voxel_size, result_global)
		'''
		
		print(register_result.transformation)
		
		# transform various pointclouds
		temp_pcd = o3d.geometry.PointCloud()
		temp_pcd.points = o3d.utility.Vector3dVector(scan_points1)
		temp_pcd = temp_pcd.transform(register_result.transformation)
		scan_points1 = np.asarray(temp_pcd.points)
		
		frame1_pcd_down = frame1_pcd_down.transform(register_result.transformation)
		
		temp_pcd = o3d.geometry.PointCloud()
		temp_pcd.points = o3d.utility.Vector3dVector(cluster_points1)
		temp_pcd = temp_pcd.transform(register_result.transformation)
		cluster_points1 = np.asarray(temp_pcd.points)
		
		'''
		# find mutual correspondances
		# nearest 2 to each 1
		dist_2near1, ind_2near1 = neighbors.NearestNeighbors().fit(cluster_points2).kneighbors(cluster_points1, n_neighbors=1, return_distance=True)
		ind_2near1 = ind_2near1[:,0]
		#print(ind_2near1)
		
		# nearest 1 to each 2
		dist_1near2, ind_1near2 = neighbors.NearestNeighbors().fit(cluster_points1).kneighbors(cluster_points2, n_neighbors=1, return_distance=True)
		ind_1near2 = ind_1near2[:,0]
		#print(ind_1near2)
		
		mut_corr = []
		for i in range(len(ind_2near1)):
			point1_idx = i
			close2_idx = ind_2near1[i]
			close1_idx = ind_1near2[close2_idx]
			if close1_idx == i:
				mut_corr.append([point1_idx, close2_idx])
		mut_corr = np.asarray(mut_corr)
		print("number of mutual correspondances: {}".format(mut_corr.shape[0]))
		
		both_points = np.concatenate((cluster_points1,cluster_points2), axis=0)
		lines_idx = mut_corr.copy()
		lines_idx[:,1] += cluster_points1.shape[0] # points2 was concatenated to the end, past points1
		corr_line_set = o3d.geometry.LineSet(
			points=o3d.utility.Vector3dVector(both_points),
			lines=o3d.utility.Vector2iVector(lines_idx),
		)
		'''
		
		# visualize
		pcd1 = o3d.geometry.PointCloud()
		pcd1.points = o3d.utility.Vector3dVector(cluster_points1)
		#pcd1.points = o3d.utility.Vector3dVector(scan_points1)
		pcd1.paint_uniform_color([1.0,0.0,0.0])
		pcd2 = o3d.geometry.PointCloud()
		pcd2.points = o3d.utility.Vector3dVector(cluster_points2)
		#pcd2.points = o3d.utility.Vector3dVector(scan_points2)
		pcd2.paint_uniform_color([0.0,0.0,1.0])
		o3d.visualization.draw_geometries([pcd1,pcd2])#,corr_line_set])
		'''
		frame1_pcd_down.paint_uniform_color([1.0,0.0,0.0])
		frame2_pcd_down.paint_uniform_color([0.0,0.0,1.0])
		o3d.visualization.draw_geometries([frame1_pcd_down,frame2_pcd_down])#,corr_line_set])
		'''
	elif mode == "compare_match_pairs_clusters_clipper_alignment":
		dso_frame_num1 = list_of_scan_indexes[i]				# list of scan indexes is not nesisarily sequential. "i" is the position in this list, not the specific index of a keyframe from DSO
		frame_id1 = out_dict["frame_id"][dso_frame_num1]
		scan_points1 = out_dict["scan_points"][dso_frame_num1]
		#frame_xyz1 = out_dict["frame_pose"][dso_frame_num1]		# according to DSO
		#w2c_matrix1 = out_dict["w2c_matrix"][dso_frame_num1]
		#c2w_kitti1 = gt_data_dict["kitti_c2w"][frame_id1]
		
		dso_frame_num2 = list_of_scan_indexes2[i]
		frame_id2 = out_dict["frame_id"][dso_frame_num2]
		scan_points2 = out_dict["scan_points"][dso_frame_num2]
		#frame_xyz2 = out_dict["frame_pose"][dso_frame_num2]		# according to DSO
		#w2c_matrix2 = out_dict["w2c_matrix"][dso_frame_num2]
		#c2w_kitti2 = gt_data_dict["kitti_c2w"][frame_id2]
		
		method_string = "cluster"
		#method_string = "pcsimp"
		
		#TODO finish conversion
		if use_semantic_labels:
			scan_labels1 = out_dict["scan_labels"][dso_frame_num1]
			scan_labels2 = out_dict["scan_labels"][dso_frame_num2]
		
		# get clusters
		if use_semantic_labels:
			cluster_dict = do_clustering({"scan_points": scan_points1,"scan_labels":scan_labels1,"target_num":TARGET_NUM, "which_clustering_override":which_clustering_override}, method=method_string, remove_noise=True, eng=eng)
		else:
			cluster_dict = do_clustering({"scan_points": scan_points1,"target_num":TARGET_NUM, "which_clustering_override":which_clustering_override}, method=method_string, remove_noise=True, eng=eng)
		cluster_points1 = cluster_dict["cluster_points"]
		if use_semantic_labels:
			cluster_dict = do_clustering({"scan_points": scan_points2,"scan_labels":scan_labels2,"target_num":TARGET_NUM, "which_clustering_override":which_clustering_override}, method=method_string, remove_noise=True, eng=eng)
		else:
			cluster_dict = do_clustering({"scan_points": scan_points2,"target_num":TARGET_NUM, "which_clustering_override":which_clustering_override}, method=method_string, remove_noise=True, eng=eng)
		cluster_points2 = cluster_dict["cluster_points"]
		'''
		# check centering
		print("diagnostic 0 = {},{}".format(cluster_points1.shape,cluster_points2.shape))
		print("diagnostic 1 = X {:.2f}:{:.2f} Y {:.2f}:{:.2f} Z {:.2f}:{:.2f}".format(
			np.amin(cluster_points1[:,0]),np.amax(cluster_points1[:,0]),
			np.amin(cluster_points1[:,1]),np.amax(cluster_points1[:,1]),
			np.amin(cluster_points1[:,2]),np.amax(cluster_points1[:,2])
		))
		print("diagnostic 2 = X {:.2f}:{:.2f} Y {:.2f}:{:.2f} Z {:.2f}:{:.2f}".format(
			np.amin(cluster_points2[:,0]),np.amax(cluster_points2[:,0]),
			np.amin(cluster_points2[:,1]),np.amax(cluster_points2[:,1]),
			np.amin(cluster_points2[:,2]),np.amax(cluster_points2[:,2])
		))
		print("diagnostic 3 = X {:.2f} Y {:.2f} Z {:.2f}".format(
			np.average(cluster_points2[:,0]),
			np.average(cluster_points2[:,1]),
			np.average(cluster_points2[:,2])
		))
		'''
		# get clusters
		#cluster_dict = do_clustering({"scan_points": scan_points1,"target_num":TARGET_NUM}, method="cluster", remove_noise=True)
		#cluster_points1 = cluster_dict["cluster_points"]
		#cluster_dict = do_clustering({"scan_points": scan_points2,"target_num":TARGET_NUM}, method="cluster", remove_noise=True)
		#cluster_points2 = cluster_dict["cluster_points"]
		
		#TODO try centering points to see if that helps
		# WARNING DOING THIS GIVES THE APPEARANCE OF ALIGNMENT BY BRINGING SETS OF CLANDMARKS CLOSER TOGETHER
		# THE CLIPPER CODE DOESN'T ACTUALLY DO ANY ALIGNMENT WITH A TRANSFORMATION MATRIX
		# THIS CAN GIVE A FALSE APPEARANCE OF SUCCESS
		#print("cluster_points1.shape: {}".format(cluster_points1.shape))
		#print("cluster_points2.shape: {}".format(cluster_points2.shape))
		#assert(0)
		#average_1 = np.average(cluster_points1, axis=0)
		#average_2 = np.average(cluster_points2, axis=0)
		#cluster_points1 = cluster_points1 - average_1
		#cluster_points2 = cluster_points2 - average_2
		'''
		print("diagnostic 0 = {},{}".format(cluster_points1.shape,cluster_points2.shape))
		print("diagnostic 1 = X {:.2f}:{:.2f} Y {:.2f}:{:.2f} Z {:.2f}:{:.2f}".format(
			np.amin(cluster_points1[:,0]),np.amax(cluster_points1[:,0]),
			np.amin(cluster_points1[:,1]),np.amax(cluster_points1[:,1]),
			np.amin(cluster_points1[:,2]),np.amax(cluster_points1[:,2])
		))
		print("diagnostic 2 = X {:.2f}:{:.2f} Y {:.2f}:{:.2f} Z {:.2f}:{:.2f}".format(
			np.amin(cluster_points2[:,0]),np.amax(cluster_points2[:,0]),
			np.amin(cluster_points2[:,1]),np.amax(cluster_points2[:,1]),
			np.amin(cluster_points2[:,2]),np.amax(cluster_points2[:,2])
		))
		print("diagnostic 3 = X {:.2f} Y {:.2f} Z {:.2f}".format(
			np.average(cluster_points2[:,0]),
			np.average(cluster_points2[:,1]),
			np.average(cluster_points2[:,2])
		))
		assert(0)
		'''
		#print("cluster_points1.shape: {} cluster_points2.shape: {}".format(cluster_points1.shape,cluster_points2.shape)) # -> (120,3)
		cluster_points1 = np.transpose(cluster_points1)
		cluster_points2 = np.transpose(cluster_points2)
		#print("cluster_points1.shape: {} cluster_points2.shape: {}".format(cluster_points1.shape,cluster_points2.shape)) # -> (3,120)
		
		def gen_A_full(points_1, points_2):
			length_1 = points_1.shape[1]
			length_2 = points_2.shape[1]
			
			temp_arr = []
			for i in range(length_1):
				for j in range(length_2):
					temp_arr.append([i,j])
			
			return np.asarray(temp_arr, dtype=np.int32)

		A = gen_A_full(cluster_points1, cluster_points2)
		print("number of possible associations fed into clipper: {}".format(A.shape[0]))
		
		#TODO subsample rows of A!
		
		#TODO couldn't find a set of tuning parameters that gave consistent success, seems sensitive to scale. (pointcloud scale or scale of inter-point distances or some property of the distribution like that)
		iparams = clipperpy.invariants.EuclideanDistanceParams()
		iparams.sigma = 0.015
		iparams.epsilon = 0.02
		invariant = clipperpy.invariants.EuclideanDistance(iparams)

		params = clipperpy.Params()
		clipper = clipperpy.CLIPPER(invariant, params)

		#(self: clipperpy.clipperpy.CLIPPER, D1: numpy.ndarray[numpy.float64[m, n]], D2: numpy.ndarray[numpy.float64[m, n]], A: numpy.ndarray[numpy.int32[m, 2]]) -> None

		t0 = time.perf_counter()
		clipper.score_pairwise_consistency(cluster_points1, cluster_points2, A)
		t1 = time.perf_counter()
		print(f"Affinity matrix creation took {t1-t0:.3f} seconds")

		t0 = time.perf_counter()
		clipper.solve()
		t1 = time.perf_counter()
		print(f"Solving took {t1-t0:.3f} seconds")

		# A = clipper.get_initial_associations()
		Ain = clipper.get_selected_associations()

		both_points = np.concatenate((cluster_points1,cluster_points2), axis=1)
		lines_idx = Ain.copy()
		lines_idx[:,1] += cluster_points1.shape[1] # points2 was concatenated to the end, past points1
		corr_line_set = o3d.geometry.LineSet(
			points=o3d.utility.Vector3dVector(both_points.T),
			lines=o3d.utility.Vector2iVector(lines_idx),
		)
		
		# visualize
		pcd1 = o3d.geometry.PointCloud()
		pcd1.points = o3d.utility.Vector3dVector(cluster_points1.T)
		pcd1.paint_uniform_color([1.0,0.0,0.0])
		#coord1 = o3d.geometry.TriangleMesh().create_coordinate_frame(size=10.0,origin=gt_data_dict["kitti_xyz"][frame_id1])
		
		pcd2 = o3d.geometry.PointCloud()
		pcd2.points = o3d.utility.Vector3dVector(cluster_points2.T)
		pcd2.paint_uniform_color([0.0,0.0,1.0])
		#coord2 = o3d.geometry.TriangleMesh().create_coordinate_frame(size=10.0,origin=gt_data_dict["kitti_xyz"][frame_id2])
		
		o3d.visualization.draw_geometries([pcd1,pcd2,corr_line_set])
	elif mode == "examine_match_pairs":
		dso_frame_num = list_of_scan_indexes[i]				# list of scan indexes is not nesisarily sequential. "i" is the position in this list, not the specific index of a keyframe from DSO
		frame_id = out_dict["frame_id"][dso_frame_num]
		scan_points = out_dict["scan_points"][dso_frame_num]
		frame_xyz = out_dict["frame_pose"][dso_frame_num]
		
		dso_frame_num2 = list_of_scan_indexes2[i]
		frame_id2 = out_dict["frame_id"][dso_frame_num2]
		scan_points2 = out_dict["scan_points"][dso_frame_num2]
		frame_xyz2 = out_dict["frame_pose"][dso_frame_num2]
		
		#use_semantic_labels = False
		
		if use_semantic_labels:
			print("assigning scan labels")
			scan_labels = out_dict["scan_labels"][dso_frame_num]
			scan_labels2 = out_dict["scan_labels"][dso_frame_num2]
		else:
			print("not assigning scan labels")
		
		#labels1,points_out1 = cluster_points(scan_points, frame_xyz)
		#labels2,points_out2 = cluster_points(scan_points2, frame_xyz2)
		
		#labels1 = label_points(scan_points)
		#labels2 = label_points(scan_points2)
		#points_out1 = scan_points
		#points_out2 = scan_points2
		
		if use_semantic_labels:
			out_dict2 = do_clustering({"scan_points":scan_points,"scan_labels":scan_labels,"target_num":TARGET_NUM, "which_clustering_override":which_clustering_override}, method="cluster", remove_noise=True)
		else:
			out_dict2 = do_clustering({"scan_points":scan_points,"target_num":TARGET_NUM, "which_clustering_override":which_clustering_override}, method="cluster", remove_noise=True)
		labels1 = out_dict2["point_labels"]
		points_out1 = out_dict2["scan_points"]
		if use_semantic_labels:
			out_dict2 = do_clustering({"scan_points":scan_points2,"scan_labels":scan_labels2,"target_num":TARGET_NUM, "which_clustering_override":which_clustering_override}, method="cluster", remove_noise=True)
		else:
			out_dict2 = do_clustering({"scan_points":scan_points2,"target_num":TARGET_NUM, "which_clustering_override":which_clustering_override}, method="cluster", remove_noise=True)
		labels2 = out_dict2["point_labels"]
		points_out2 = out_dict2["scan_points"]
		
		print("number of clusters: 1:{} 2:{}".format(len(np.unique(labels1))-1,len(np.unique(labels2))-1))
		
		point_colors1 = colors_list[((labels1+1)%colors_list.shape[0]),:]
		point_colors1[labels1 == -1] = [0,0,0]
		point_colors2 = colors_list[((labels2+1)%colors_list.shape[0]),:]
		point_colors2[labels2 == -1] = [0,0,0]
		
		pcd1 = o3d.geometry.PointCloud()
		pcd1.points = o3d.utility.Vector3dVector(points_out1)
		pcd1.colors = o3d.utility.Vector3dVector(point_colors1)
		coord1 = o3d.geometry.TriangleMesh().create_coordinate_frame(size=10.0,origin=frame_xyz)
		
		pcd2 = o3d.geometry.PointCloud()
		pcd2.points = o3d.utility.Vector3dVector(points_out2)
		pcd2.colors = o3d.utility.Vector3dVector(point_colors2)
		coord2 = o3d.geometry.TriangleMesh().create_coordinate_frame(size=10.0,origin=frame_xyz2)
		
		vis.clear_geometries()
		vis2.clear_geometries()
		vis.add_geometry(coord1)
		vis.add_geometry(pcd1)
		vis2.add_geometry(coord2)
		vis2.add_geometry(pcd2)
		vis.reset_view_point(True)
		vis2.reset_view_point(True)
		try:
			while True:
				if not vis.poll_events():
					#vis2.destroy_window()
					#vis.destroy_window()
					break
				vis.update_renderer()
				
				if not vis2.poll_events():
					#vis.destroy_window()
					#vis2.destroy_window()
					break
				vis2.update_renderer()
		except KeyboardInterrupt:
			pass
		
	elif mode == "examine_clusters_sequence":
		dso_frame_num = list_of_scan_indexes[i]				# list of scan indexes is not nesisarily sequential. "i" is the position in this list, not the specific index of a keyframe from DSO
		frame_id = out_dict["frame_id"][dso_frame_num]
		scan_points = out_dict["scan_points"][dso_frame_num]
		frame_xyz = out_dict["frame_pose"][dso_frame_num]
		
		#my_points = scan_points
		
		#pcd = o3d.geometry.PointCloud()
		#pcd.points = o3d.utility.Vector3dVector(my_points)
		#cloud, ind = pcd.remove_radius_outlier(nb_points=2, radius=2.5)
		#my_points = np.take(my_points, ind, axis=0)

		#labels = label_points(my_points)
		out_dict2 = do_clustering({"scan_points":scan_points,"target_num":TARGET_NUM, "which_clustering_override":which_clustering_override}, method="cluster", remove_noise=True)
		labels = out_dict2["point_labels"]
		my_points = out_dict2["scan_points"]
		
		print("number of clusters: {}".format(len(np.unique(labels))-1))
		
		point_colors = colors_list[((labels+1)%colors_list.shape[0]),:]
		point_colors[labels == -1] = [0,0,0]
		
		pcd = o3d.geometry.PointCloud()
		pcd.points = o3d.utility.Vector3dVector(my_points)
		pcd.colors = o3d.utility.Vector3dVector(point_colors)
		#pcd.paint_uniform_color([0,0,1.0])
		coord1 = o3d.geometry.TriangleMesh().create_coordinate_frame(size=10.0,origin=frame_xyz)
		o3d.visualization.draw_geometries([pcd,coord1])
		
	elif mode == "test_noisy_point_removal":
		dso_frame_num = list_of_scan_indexes[i]				# list of scan indexes is not nesisarily sequential. "i" is the position in this list, not the specific index of a keyframe from DSO
		frame_id = out_dict["frame_id"][dso_frame_num]
		scan_points = out_dict["scan_points"][dso_frame_num]
		frame_xyz = out_dict["frame_pose"][dso_frame_num]
		
		pcd = o3d.geometry.PointCloud()
		pcd.points = o3d.utility.Vector3dVector(scan_points)
		cloud, ind = pcd.remove_radius_outlier(nb_points=2, radius=2.5)

		print("points shape: {} cloud shape: {} ind shape: {}".format(scan_points.shape, np.asarray(cloud).shape, np.asarray(ind).shape))
		#inlier_cloud = np.take(scan_points, np.asarray(ind)-1, axis=0)
		
		cloud_colors = np.repeat([[0.0,0.0,1.0]], scan_points.shape[0], axis=0)
		cloud_colors[ind] = [1.0,0.0,0.0]
		pcd.colors = o3d.utility.Vector3dVector(cloud_colors)
		coord1 = o3d.geometry.TriangleMesh().create_coordinate_frame(size=10.0,origin=frame_xyz)
		o3d.visualization.draw_geometries([pcd,coord1])
		
	elif mode == "examine_clusters_points_pairs":
		dso_frame_num = list_of_scan_indexes[i]				# list of scan indexes is not nesisarily sequential. "i" is the position in this list, not the specific index of a keyframe from DSO
		frame_id = out_dict["frame_id"][dso_frame_num]
		scan_points = out_dict["scan_points"][dso_frame_num]
		frame_xyz = out_dict["frame_pose"][dso_frame_num]
		
		dso_frame_num2 = list_of_scan_indexes2[i]
		frame_id2 = out_dict["frame_id"][dso_frame_num2]
		scan_points2 = out_dict["scan_points"][dso_frame_num2]
		frame_xyz2 = out_dict["frame_pose"][dso_frame_num2]
		
		points_out1 = do_clustering({"scan_points":scan_points,"target_num":TARGET_NUM, "which_clustering_override":which_clustering_override}, method="cluster", remove_noise=True)["cluster_points"]
		points_out2 = do_clustering({"scan_points":scan_points2,"target_num":TARGET_NUM, "which_clustering_override":which_clustering_override}, method="cluster", remove_noise=True)["cluster_points"]
		print("number of clusters: 1:{} 2:{}".format(points_out1.shape[0],points_out2.shape[0]))
		
		pcd1 = o3d.geometry.PointCloud()
		pcd1.points = o3d.utility.Vector3dVector(points_out1)
		pcd1.paint_uniform_color([0,0.5,1.0])
		coord1 = o3d.geometry.TriangleMesh().create_coordinate_frame(size=10.0,origin=frame_xyz)
		
		pcd2 = o3d.geometry.PointCloud()
		pcd2.points = o3d.utility.Vector3dVector(points_out2)
		pcd2.paint_uniform_color([0,0.5,1.0])
		coord2 = o3d.geometry.TriangleMesh().create_coordinate_frame(size=10.0,origin=frame_xyz2)
		
		vis.clear_geometries()
		vis2.clear_geometries()
		vis.add_geometry(coord1)
		vis.add_geometry(pcd1)
		vis2.add_geometry(coord2)
		vis2.add_geometry(pcd2)
		vis.reset_view_point(True)
		vis2.reset_view_point(True)
		try:
			while True:
				if not vis.poll_events():
					#vis2.destroy_window()
					#vis.destroy_window()
					break
				vis.update_renderer()
				
				if not vis2.poll_events():
					#vis.destroy_window()
					#vis2.destroy_window()
					break
				vis2.update_renderer()
		except KeyboardInterrupt:
			pass
	elif mode == "compare_match_pairs_clusters_alignment_measurement":
		
		dso_frame_num1 = list_of_scan_indexes[i]				# list of scan indexes is not nesisarily sequential. "i" is the position in this list, not the specific index of a keyframe from DSO
		frame_id1 = out_dict["frame_id"][dso_frame_num1]
		scan_points1 = out_dict["scan_points"][dso_frame_num1]
		#frame_xyz1 = out_dict["frame_pose"][dso_frame_num1]		# according to DSO
		w2c_matrix1 = out_dict["w2c_matrix"][dso_frame_num1]
		c2w_kitti1 = gt_data_dict["kitti_c2w"][frame_id1]
		
		dso_frame_num2 = list_of_scan_indexes2[i]
		frame_id2 = out_dict["frame_id"][dso_frame_num2]
		scan_points2 = out_dict["scan_points"][dso_frame_num2]
		#frame_xyz2 = out_dict["frame_pose"][dso_frame_num2]		# according to DSO
		w2c_matrix2 = out_dict["w2c_matrix"][dso_frame_num2]
		c2w_kitti2 = gt_data_dict["kitti_c2w"][frame_id2]
		
		# selection of clustering method
				
		method_string = "cluster"
		#method_string = "pcsimp"
		
		#which_clustering_override = None
		#which_clustering_override = "birch"
		#which_clustering_override = "dbscan"
		#which_clustering_override = args.which_clustering_override			#TODO perhaps add to the other methods?
		
		#use_semantic_labels = False
		
		if use_semantic_labels:
			scan_labels1 = out_dict["scan_labels"][dso_frame_num1]
			scan_labels2 = out_dict["scan_labels"][dso_frame_num2]
		
		#TODO note: have already added the ability to disable correction of the total number of clusters to this mode specifically, below (correct_num_clusters) #TODO could add this to other modes, like seems to have been done with "which clustering override"
		# get clusters
		if use_semantic_labels:
			cluster_dict = do_clustering({"scan_points": scan_points1,"scan_labels":scan_labels1,"target_num":TARGET_NUM, "which_clustering_override":which_clustering_override}, method=method_string, remove_noise=True, eng=eng, correct_num_clusters=correct_num_clusters)
		else:
			cluster_dict = do_clustering({"scan_points": scan_points1,"target_num":TARGET_NUM, "which_clustering_override":which_clustering_override}, method=method_string, remove_noise=True, eng=eng, correct_num_clusters=correct_num_clusters)
		cluster_points1 = cluster_dict["cluster_points"]
		
		if use_semantic_labels:
			cluster_dict = do_clustering({"scan_points": scan_points2,"scan_labels":scan_labels2,"target_num":TARGET_NUM, "which_clustering_override":which_clustering_override}, method=method_string, remove_noise=True, eng=eng, correct_num_clusters=correct_num_clusters)
		else:
			cluster_dict = do_clustering({"scan_points": scan_points2,"target_num":TARGET_NUM, "which_clustering_override":which_clustering_override}, method=method_string, remove_noise=True, eng=eng, correct_num_clusters=correct_num_clusters)
		cluster_points2 = cluster_dict["cluster_points"]
		
		# transfer from keyframe coords to global coord using gt (simulating grass alignment)
		scanpts1_h = np.concatenate((cluster_points1, np.ones((cluster_points1.shape[0],1))), axis=1)
		scanpts1_cam_h = np.matmul(w2c_matrix1,np.transpose(scanpts1_h))
		scanpts1_global_h = np.matmul(c2w_kitti1,scanpts1_cam_h)
		scanpts1_global = np.transpose(scanpts1_global_h)[:,0:3]
		
		scanpts2_h = np.concatenate((cluster_points2, np.ones((cluster_points2.shape[0],1))), axis=1)
		scanpts2_cam_h = np.matmul(w2c_matrix2,np.transpose(scanpts2_h))
		scanpts2_global_h = np.matmul(c2w_kitti2,scanpts2_cam_h)
		scanpts2_global = np.transpose(scanpts2_global_h)[:,0:3]
		
		# dict to record lists of statistics over the sequence
		#results_recording_dict = {
		#	"frame_1_max":[],			# the maximum limits of points in the frame
		#	"frame_1_min":[],
		#	"frame_2_max":[],
		#	"frame_2_min":[],
		#	"num_points_1":[],			# the number of points in the frame. together with the bounds of the bounding box above, rough density can be computed
		#	"num_points_2":[],
		#	
		#	"num_clusters" : [],		# should be the same for both frames, technically for all frames
		#	"num_outliers_nn" : [],		# number of cluster points not in a mutual-nn relationship. should be same between sets, see below
		#	"dists_nearest_neighbors" : [],	# for those in a nn relationship, how far apart are they?
		#	
		#	"dists_to_closest" : []		# record all the distance to the nearest point for all points in frame 1, we can apply range gating later, or just plot a distribution
		#								# we only record the number of outliers that don't have a pairing within 1m in frame 1, we'll get to the reverse pairing where we consider frame 2 later in the sequence
		#}
		
		# find mutual correspondances
		# nearest 2 to each 1
		dist_2near1, ind_2near1 = neighbors.NearestNeighbors().fit(scanpts2_global).kneighbors(scanpts1_global, n_neighbors=1, return_distance=True)
		ind_2near1 = ind_2near1[:,0]
		#print(ind_2near1)
		
		# nearest 1 to each 2
		dist_1near2, ind_1near2 = neighbors.NearestNeighbors().fit(scanpts1_global).kneighbors(scanpts2_global, n_neighbors=1, return_distance=True)
		ind_1near2 = ind_1near2[:,0]
		#print(ind_1near2)
		
		
		mut_corr = []
		for i in range(len(ind_2near1)):
			point1_idx = i
			close2_idx = ind_2near1[i]
			close1_idx = ind_1near2[close2_idx]
			if close1_idx == i:
				mut_corr.append([point1_idx, close2_idx])
		mut_corr = np.asarray(mut_corr)
		print("number of mutual correspondances: {}".format(mut_corr.shape[0]))
		
		
		'''
		print(scanpts1_global.shape)
		max_vec = np.amax(scanpts1_global, axis=0)
		min_vec = np.amin(scanpts1_global, axis=0)
		width_1 = max_vec-min_vec
		print("{} {}".format(max_vec,min_vec))
		print(width_1)
		max_vec = np.amax(scanpts2_global, axis=0)
		min_vec = np.amin(scanpts2_global, axis=0)
		width_2 = max_vec-min_vec
		print("{} {}".format(max_vec,min_vec))
		print(width_2)
		assert(0)
		'''
		
		# record some stats about the size and number of points useful for determining density
		results_recording_dict["frame_1_max"].append(np.amax(scan_points1, axis=0))
		results_recording_dict["frame_1_min"].append(np.amin(scan_points1, axis=0))
		results_recording_dict["frame_2_max"].append(np.amax(scan_points2, axis=0))
		results_recording_dict["frame_2_min"].append(np.amin(scan_points2, axis=0))
		results_recording_dict["num_points_1"].append(scan_points1.shape[0])
		results_recording_dict["num_points_2"].append(scan_points2.shape[0])
		
		# attempting to use pyntcloud to figure out max density. doesn't seem to work right from colab testing
		'''
		extents = np.amax(scanpts1_global, axis=0)-np.amin(scanpts1_global, axis=0)
		print(extents)
		extents = extents.astype(int)
		print(extents)
		assert(0)
		points = pd.DataFrame(scanpts1_global)
		cloud = PyntCloud(pd.DataFrame(data=points, columns=["x", "y", "z"]))
		voxelgrid_id = cloud.add_structure("voxelgrid", n_x=90, n_y=10, n_z=90) #TODO set number of division based on length/width/height (extent), not clear how division works, need proof of concept in a test colab notebook
		#voxelgrid_id = cloud.add_structure("voxelgrid", size_x=10.0, size_y=10.0, size_z=10.0)
		voxelgrid = cloud.structures[voxelgrid_id]								# We use the calculated occupied voxel grid ids to create the voxel representation of the point cloud
		density_feature_vector = voxelgrid.get_feature_vector(mode="density")	# We extract the density feature for each occupied voxel that we will use for coloring the voxels
		max_density = density_feature_vector.max()
		print("max density: {} {}".format(max_density, max_density*scanpts1_global.shape[0]))
		min_density = density_feature_vector.min()
		print("min density: {} {}".format(min_density, min_density*scanpts1_global.shape[0]))
		avg_density = np.average(density_feature_vector)
		print("avg density: {} {}".format(avg_density, avg_density*scanpts1_global.shape[0]))
		'''
		
		# record some stats about the size and number of points useful for determining density
		
		results_recording_dict["num_clusters"].append(scanpts1_global.shape[0])

		results_recording_dict["num_outliers_nn"].append(scanpts1_global.shape[0]-mut_corr.shape[0])
		mnn_dists = []
		for i in range(len(mut_corr)):
			point1_idx = mut_corr[i,0]
			point2_idx = mut_corr[i,1]
			dist = np.linalg.norm(scanpts1_global[point1_idx]-scanpts2_global[point2_idx])
			mnn_dists.append(dist)
		results_recording_dict["dists_nearest_neighbors"].append(mnn_dists)
		
		
		# compute dists matrix points1 x points2
		# find the minimum in each row -> record all
		dists_matrix = scipy.spatial.distance.cdist(scanpts1_global, scanpts2_global, metric='euclidean')
		#min_index_in_each_row = np.argmin(dists_matrix, axis=1)
		minvals = np.amin(dists_matrix, axis=1)
		results_recording_dict["dists_to_closest"].append(minvals)
		

		'''
		#pulled from experimentation colab, may need adaptation for use
		# if that minimum is less than limit -> add one to rangegate count, record
		distance_limit = 1
		num_dists_under_limit = 0
		dists_under_limit = []
		for i in range(len(minvals)):
			if minvals[i] < distance_limit:
				num_dists_under_limit += 1
				dists_under_limit.append(minvals[i])
		dists_under_limit = np.asarray(dists_under_limit)

		print("number under distance limit: {}".format(num_dists_under_limit))
		print("minimums under distance limit: {}".format(dists_under_limit))
		'''
		
		
		both_points = np.concatenate((scanpts1_global,scanpts2_global), axis=0)
		lines_idx = mut_corr.copy()
		lines_idx[:,1] += scanpts1_global.shape[0] # points2 was concatenated to the end, past points1
		corr_line_set = o3d.geometry.LineSet(
			points=o3d.utility.Vector3dVector(both_points),
			lines=o3d.utility.Vector2iVector(lines_idx),
		)
		
		#TO/DO add visualization option for range gating, for tuning
		if args.skip_vis != True:
			# visualize
			pcd1 = o3d.geometry.PointCloud()
			pcd1.points = o3d.utility.Vector3dVector(scanpts1_global)
			pcd1.paint_uniform_color([1.0,0.0,0.0])
			coord1 = o3d.geometry.TriangleMesh().create_coordinate_frame(size=10.0,origin=gt_data_dict["kitti_xyz"][frame_id1])
			
			pcd2 = o3d.geometry.PointCloud()
			pcd2.points = o3d.utility.Vector3dVector(scanpts2_global)
			pcd2.paint_uniform_color([0.0,0.0,1.0])
			coord2 = o3d.geometry.TriangleMesh().create_coordinate_frame(size=10.0,origin=gt_data_dict["kitti_xyz"][frame_id2])
			
			o3d.visualization.draw_geometries([pcd1,pcd2,coord1,coord2,corr_line_set])
		
		
			# dict to record lists of statistics over the sequence
		#results_recording_dict = {
		#	"frame_1_max":[],			# the maximum limits of points in the frame
		#	"frame_1_min":[],
		#	"frame_2_max":[],
		#	"frame_2_min":[],
		#	"num_points_1":[],			# the number of points in the frame. together with the bounds of the bounding box above, rough density can be computed
		#	"num_points_2":[],
		#	
		#	"num_clusters" : [],		# should be the same for both frames, technically for all frames
		#	"num_outliers_nn" : [],		# number of cluster points not in a mutual-nn relationship. should be same between sets, see below
		#	"dists_nearest_neighbors" : [],	# for those in a nn relationship, how far apart are they?
		#	
		#	"dists_to_closest" : []		# record all the distance to the nearest point for all points in frame 1, we can apply range gating later, or just plot a distribution
		#								# we only record the number of outliers that don't have a pairing within 1m in frame 1, we'll get to the reverse pairing where we consider frame 2 later in the sequence
		#}

	elif mode == "test_real_and_synthetic_landmark_association":
		# get clusters (one set)
		dso_frame_num1 = list_of_scan_indexes[i]				# list of scan indexes is not nesisarily sequential. "i" is the position in this list, not the specific index of a keyframe from DSO
		frame_id1 = out_dict["frame_id"][dso_frame_num1]
		scan_points1 = out_dict["scan_points"][dso_frame_num1]
		#frame_xyz1 = out_dict["frame_pose"][dso_frame_num1]		# according to DSO
		w2c_matrix1 = out_dict["w2c_matrix"][dso_frame_num1]
		c2w_kitti1 = gt_data_dict["kitti_c2w"][frame_id1]
		
		#dso_frame_num2 = list_of_scan_indexes2[i]
		#frame_id2 = out_dict["frame_id"][dso_frame_num2]
		#scan_points2 = out_dict["scan_points"][dso_frame_num2]
		##frame_xyz2 = out_dict["frame_pose"][dso_frame_num2]		# according to DSO
		#w2c_matrix2 = out_dict["w2c_matrix"][dso_frame_num2]
		#c2w_kitti2 = gt_data_dict["kitti_c2w"][frame_id2]
		
		#--------------------------------------------------------------------------------------------
		# only one set of clusters is needed, as the other will be generated from it in a controlled way for testing
		
		
		# selection of clustering method
		#method_string = "cluster"
		#which_clustering_override = None
		#which_clustering_override = "birch"
		#which_clustering_override = "dbscan"
		#which_clustering_override = args.which_clustering_override
		#use_semantic_labels = False
		
		if use_semantic_labels:
			scan_labels1 = out_dict["scan_labels"][dso_frame_num1]
		#	scan_labels2 = out_dict["scan_labels"][dso_frame_num2]
		
		# get clusters
		if use_semantic_labels:
			cluster_dict = do_clustering({"scan_points": scan_points1,"scan_labels":scan_labels1,"target_num":TARGET_NUM, "which_clustering_override":which_clustering_override}, method="cluster", remove_noise=True, eng=eng)
		else:
			cluster_dict = do_clustering({"scan_points": scan_points1,"target_num":TARGET_NUM, "which_clustering_override":which_clustering_override}, method="cluster", remove_noise=True, eng=eng)
		cluster_points1 = cluster_dict["cluster_points"]
		
		'''
		if use_semantic_labels:
			cluster_dict = do_clustering({"scan_points": scan_points2,"scan_labels":scan_labels2,"target_num":TARGET_NUM, "which_clustering_override":which_clustering_override}, method="cluster", remove_noise=True, eng=eng)
		else:
			cluster_dict = do_clustering({"scan_points": scan_points2,"target_num":TARGET_NUM, "which_clustering_override":which_clustering_override}, method="cluster", remove_noise=True, eng=eng)
		cluster_points2 = cluster_dict["cluster_points"]
		'''
		
		'''
		# transfer from keyframe coords to global coord using gt (simulating grass alignment)
		scanpts1_h = np.concatenate((cluster_points1, np.ones((cluster_points1.shape[0],1))), axis=1)
		scanpts1_cam_h = np.matmul(w2c_matrix1,np.transpose(scanpts1_h))
		scanpts1_global_h = np.matmul(c2w_kitti1,scanpts1_cam_h)
		scanpts1_global = np.transpose(scanpts1_global_h)[:,0:3]
		
		scanpts2_h = np.concatenate((cluster_points2, np.ones((cluster_points2.shape[0],1))), axis=1)
		scanpts2_cam_h = np.matmul(w2c_matrix2,np.transpose(scanpts2_h))
		scanpts2_global_h = np.matmul(c2w_kitti2,scanpts2_cam_h)
		scanpts2_global = np.transpose(scanpts2_global_h)[:,0:3]
		'''
		
		# random rotation is about the origin, so points need to be moved there
		# other operations only add offsets or are within the existing bounds of the point set
		
		# trying to center the points, poorly
		#cluster_points1 = cluster_points1 - (np.amin(cluster_points1, axis=0)+np.amax(cluster_points1, axis=0))/2.0
		
		mean_coord = np.mean(cluster_points1, axis=0)
		scan_points1 = scan_points1 - mean_coord
		cluster_points1 = cluster_points1 - mean_coord
		#print("mean_coord: {}".format(mean_coord))
		
		#--------------------------------------------------------------------------------------------
		
		# needs to use cluster points from a clustering technique, not raw scan points
		# regular birch is probably best clustering technique, though perhaps multiple should be tested (slightly different distribution of clusters)
		
		point_set_1 = cluster_points1
		point_set_2 = cluster_points1
		
		rng = np.random.default_rng(5)
		
		# generate corresponding set with appropriate outliers and noise
		if args.gen_pcent_ouliers is None:
			assert 0, "gen_pcent_ouliers cannot be None in this mode"
		if args.std_dev_noise is None:
			assert 0, "std_dev_noise cannot be None in this mode"
		
		#TODO set up command line arg control for these
		do_rotation = True
		#do_rotation = False
		do_translation = True
		#do_translation = False
		
		# these default to tru and are controlled through commandline args
		# setting the options to zero on the commandline nullifies thier effect
		do_outliers = True
		do_noise = True
		
		'''
		# rotation
		if do_rotation:
			point_set_2 = uniform_random_rotation_helper(point_set_2, rng)

		# translation
		max_displacement = 45
		v = rng.random((1,3))
		random_offset = (v / np.linalg.norm(v))*rng.random()*max_displacement
		if do_translation:
			point_set_2 = point_set_2 + random_offset
		'''
		
		#create rotation and translation matrices (row-major order)
		#-----------------------------------------
		# https://towardsdatascience.com/the-one-stop-guide-for-transformation-matrices-cea8f609bdb1
		# https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/geometry/row-major-vs-column-major-vector.html

		# rotation matrix
		rot_matrix = uniform_random_rotation_matrix_helper(rng)
		rot_t = np.concatenate((
		  np.concatenate((rot_matrix,np.asarray([
			[0],
			[0],
			[0]
		  ])), axis=1),
		  np.asarray([[0,0,0,1]])
		  ), axis=0
		)
		#print("rot_t:\n{}".format(rot_t))

		# translation matrix
		trans_vector = rng.uniform(low=[-45.0, -45.0, 0.0], high=[45.0, 45.0, 8.0], size=(1,3)) #TODO should be 45,8,45?
		trans_t = np.identity(4)
		trans_t[3,0:3] = trans_vector
		#print("trans_t:\n{}".format(trans_t))
		#-----------------------------------------
		if not do_rotation:
			rot_t = np.identity(4)
		if not do_translation:
			trans_t = np.identity(4)
		
		#compose matrices, transform point_set_2 to get transformed point_set_2
		composed_transform_matrix = np.dot(rot_t,trans_t)
		
		points_temp = np.concatenate((point_set_2,np.ones((point_set_2.shape[0],1))), axis=1)
		points_temp = np.dot(points_temp,composed_transform_matrix)
		point_set_2 = points_temp[:,0:3]
		
		# conceptually:
		#points_rot = np.dot(point_set_2,rot_t,trans_t) # take points, rotate them, then translate
		#-----------------------------------------
		
		# create a clean set of points 2 without noise or outliers for use when checking alignment
		clean_point_set_2 = np.copy(point_set_2)
		
		#TODO DONT USE THIS, USE THE BLOCK BELOW
		'''
		# randomly replace with outliers
		old_count = point_set_2.shape[0]
		remove_num = math.floor(old_count*args.gen_pcent_ouliers)
		lower_xyz_bound = np.amin(point_set_2, axis=0)
		upper_xyz_bound = np.amax(point_set_2, axis=0)
		del_sel = np.sort(rng.choice(old_count,size=remove_num,replace=False))
		remaining = np.delete(point_set_2, del_sel, axis=0)
		new_points = rng.uniform(low=lower_xyz_bound, high=upper_xyz_bound, size=(remove_num,3))
		if do_outliers:
			point_set_2 = np.concatenate((remaining,new_points), axis=0)
		'''
		
		# randomly replace with outliers, preserve order
		old_count = point_set_2.shape[0]
		remove_num = math.floor(old_count*args.gen_pcent_ouliers)
		#print("num outliers to generate: {}".format(remove_num))
		rand_mask = np.full(old_count, True)
		rand_mask[:remove_num] = False
		rng.shuffle(rand_mask)
		lower_xyz_bound = np.amin(point_set_2, axis=0)
		upper_xyz_bound = np.amax(point_set_2, axis=0)
		all_outliers = rng.uniform(low=lower_xyz_bound, high=upper_xyz_bound, size=(old_count,3))
		if do_outliers:
			point_set_2 = np.where(rand_mask, point_set_2.transpose(), all_outliers.transpose()).transpose()
		
		#TODO DONT USE THIS, USE THE BLOCK BELOW
		'''
		# randomly replace with outliers, outside initial shape
		#TODO alternative approach to outliers, placing them outside the pointclouds
		old_count = point_set_2.shape[0]
		remove_num = math.floor(old_count*args.gen_pcent_ouliers)
		#print("num outliers to generate: {}".format(remove_num))
		lower_xyz_bound = np.amin(point_set_2, axis=0)
		upper_xyz_bound = np.amax(point_set_2, axis=0)
		del_sel = np.sort(rng.choice(old_count,size=remove_num,replace=False))
		remaining = np.delete(point_set_2, del_sel, axis=0)
		#new_points = rng.uniform(low=lower_xyz_bound, high=upper_xyz_bound, size=(remove_num,3))
		diagonal_dist = np.linalg.norm(upper_xyz_bound-lower_xyz_bound)
		radius = 0.5*diagonal_dist
		unit_vecs = np.full((remove_num,3), fill_value=[1.0,0.0,0.0])
		#print("unit_vecs.shape: {}".format(unit_vecs.shape))
		if unit_vecs.shape[0] > 0:
			rotated_vecs = np.asarray([uniform_random_rotation_helper(vec, rng)[0] for vec in unit_vecs])
		else:
			# if there aren't any vectors, don't bother trying to rotate them
			rotated_vecs = unit_vecs
		#print("rotated_vecs.shape: {}".format(rotated_vecs.shape))
		#offset_vecs = rotated_vecs*rng.normal(loc=0.0, scale=noise_std_dev, size=(rotated_vecs.shape[0],1))
		#offset_vecs = rotated_vecs*scipy.stats.halfnorm.rvs(scale=noise_std_dev, size=(rotated_vecs.shape[0],1))
		new_points = rotated_vecs*(3*radius) # produce a bunch of outliers somewhere outside the pointclouds being aligned
		#print("new_points.shape: {}".format(new_points.shape))
		if do_outliers:
			point_set_2 = np.concatenate((remaining,new_points), axis=0)
		'''
		
		#TODO #TODO #TODO #TODO #TODO #TODO #TODO #TODO #TODO #TODO #TODO
		'''
		# this is for "gentler outlier" remember to remove this later and reinstate the correct paragraph above
		# randomly replace with outliers, outside initial shape, preserve order
		old_count = point_set_2.shape[0]
		remove_num = math.floor(old_count*args.gen_pcent_ouliers)
		#print("num outliers to generate: {}".format(remove_num))
		rand_mask = np.full(old_count, True)
		rand_mask[:remove_num] = False
		rng.shuffle(rand_mask)
		lower_xyz_bound = np.amin(point_set_2, axis=0)
		upper_xyz_bound = np.amax(point_set_2, axis=0)
		diagonal_dist = np.linalg.norm(upper_xyz_bound-lower_xyz_bound)
		radius = 0.5*diagonal_dist
		unit_vecs = np.full((old_count,3), fill_value=[1.0,0.0,0.0])
		if unit_vecs.shape[0] > 0:
			rotated_vecs = np.asarray([uniform_random_rotation_helper(vec, rng)[0] for vec in unit_vecs])
		else:
			rotated_vecs = unit_vecs
		all_outliers = rotated_vecs*(3*radius) # produce a bunch of outliers somewhere outside the shape being aligned
		if do_outliers:
			point_set_2 = np.where(rand_mask, point_set_2.transpose(), all_outliers.transpose()).transpose()
		assert(args.gen_pcent_ouliers > 0.0)
		'''
		#TODO unused
		# add random noise (multiplicative, gaussian)
		#mul_noise = rng.normal(loc=1.0, scale=noise_std_dev, size=point_set_2.shape)
		#point_set_2 = np.multiply(point_set_2, mul_noise)
		
		#TODO DONT USE THIS, USE THE BLOCK BELOW -------> is there actually any difference? I don't think so be we replaced it at the same time as the order-preserving outlier code blocks above
		'''
		# add random noise (offset, gaussian)
		unit_vecs = np.full((point_set_2.shape[0],3), fill_value=[1.0,0.0,0.0])
		rotated_vecs = np.asarray([uniform_random_rotation_helper(vec, rng)[0] for vec in unit_vecs])
		#offset_vecs = rotated_vecs*rng.normal(loc=0.0, scale=args.std_dev_noise, size=(rotated_vecs.shape[0],1))
		offset_vecs = rotated_vecs*scipy.stats.halfnorm.rvs(scale=args.std_dev_noise, size=(rotated_vecs.shape[0],1))
		if do_noise:
			point_set_2 = point_set_2 + offset_vecs
		'''
		
		# add random noise (offset, gaussian)
		unit_vecs = np.full((point_set_2.shape[0],3), fill_value=[1.0,0.0,0.0])
		rotated_vecs = np.asarray([uniform_random_rotation_helper(vec, rng)[0] for vec in unit_vecs])
		#offset_vecs = rotated_vecs*rng.normal(loc=0.0, scale=args.std_dev_noise, size=(rotated_vecs.shape[0],1))
		offset_vecs = rotated_vecs*scipy.stats.halfnorm.rvs(scale=args.std_dev_noise, size=(rotated_vecs.shape[0],1))
		if do_noise:
			point_set_2 = point_set_2 + offset_vecs
		
		point_set_2 = np.ascontiguousarray(point_set_2)
		#--------------------------------------------------------------------------------------------
		
		# run grassmannian, measure accuracy of transform and number of associations found
		out_dict2 = eng.grassgraph_assoc_test({"X":matlab.double(point_set_1),"Y":matlab.double(point_set_2),"p":grass_parameters})
		
		rA = np.asarray(out_dict2["rA"])
		num_matches = out_dict2["num_matches"]
		#print("rA matrix:\n{}".format(rA))
		print("num_matches: {} out of {}".format(num_matches, point_set_1.shape[0]))
		
		'''
		# Y = X*rA
		# pts2 = pts1*rA
		'''
		
		# transform point_set_1 to match point_set_2
		point_set_1_h = np.concatenate((point_set_1,np.ones((point_set_1.shape[0],1))), axis=1)
		point_set_1_h_aligned = np.matmul(point_set_1_h,rA)
		point_set_1_aligned = point_set_1_h_aligned[:,0:3]
		
		corrX = np.asarray(out_dict2["corrX"])
		corrY = np.asarray(out_dict2["corrY"])
		
		#XrA = np.asarray(out_dict2["XrA"]) 	# XrA and corrX_aligned are the same thing
		
		# transform corr.X to match corr.Y
		corrX_h = np.concatenate((corrX,np.ones((corrX.shape[0],1))), axis=1)
		corrX_h_aligned = np.matmul(corrX_h,rA)
		corrX_aligned = corrX_h_aligned[:,0:3]
		
		#print("corrX.shape: {}".format(corrX.shape))
		#print("corrY.shape: {}".format(corrY.shape))
		#print("XrA.shape: {}".format(XrA.shape))
		
		#XrA = corr.X*rA
		#temp1 = XrA-corrX			#-> not alligned well
		#temp2 = XrA-corrX_aligned	#-> zero
		#print("diff:\n{}".format(temp1))
		#print("diff:\n{}".format(temp2))
		
		# transform scan_points1 using A
		scan_points1_h = np.concatenate((scan_points1,np.ones((scan_points1.shape[0],1))), axis=1)
		scan_points2_h = np.matmul(scan_points1_h,composed_transform_matrix)
		scan_points2 = scan_points2_h[:,0:3]
		
		# transform scan_points1 using rA
		scan_points1_h = np.concatenate((scan_points1,np.ones((scan_points1.shape[0],1))), axis=1)
		scan_points1_h_aligned = np.matmul(scan_points1_h,rA)
		scan_points1_aligned = scan_points1_h_aligned[:,0:3]
		
		#--------------------------------------------------------------------------------------------
		# run other assoc methods
		# TODO
		
		
		#--------------------------------------------------------------------------------------------
		# collect measures and write out result line (for these outlier and noise settings) for later summarization
		
		# num outliers
		# closeness of inlier points
			# transfer corr.x and corr.y
		# closeness of all points, from transform
		
		# when checking alignment quality, use a copy of the point set from before outliers and noise are added (clean_point_set_2)
		# set 1 or corr.X are always the ones transformed to match set 2 or corr.Y
		# set 2 is the one that has noise and outliers applied, so set 1 should match the clean copy of set 2 perfectly when transformed correctly
		#print("")
		#print("point_set_1_aligned.shape: {}".format(point_set_1_aligned.shape))
		#print("clean_point_set_2.shape: {}".format(clean_point_set_2.shape))
		#print("corrY.shape: {}".format(corrY.shape))
		#print("corrX_aligned.shape: {}".format(corrX_aligned.shape))
		
		#point_dists_inliers = np.linalg.norm(clean_point_set_2-point_set_1_aligned,axis=1)
		#point_dists_fullset = np.linalg.norm(corrY-corrX_aligned,axis=1)
		#print("point_dists_matches.shape: {}".format(point_dists_matches.shape))
		#print("point_dists_fullset.shape: {}".format(point_dists_fullset.shape))
		
		#print("point_dists_inliers.shape: {}".format(point_dists_inliers.shape))
		#print("point_dists_fullset.shape: {}".format(point_dists_fullset.shape))
		#print("(clean_point_set_2-point_set_1_aligned).shape: {}".format((clean_point_set_2-point_set_1_aligned).shape))
		#print("(corrY-corrX_aligned).shape: {}".format((corrY-corrX_aligned).shape))
		
		#TODO don't use, very wrong, only preserved for posterity
		frob_inliers = np.linalg.norm((clean_point_set_2-point_set_1_aligned),'fro')
		frob_fullset = np.linalg.norm((corrY-corrX_aligned),'fro')
		#--------------------------------------------------------------------------------------------
		# frob between recovered and initial transformation matrix
		
		matrix_frob = np.linalg.norm((rA-composed_transform_matrix),'fro')
		#print("frob_norm: {}".format(matrix_frob))
		
		#--------------------------------------------------------------------------------------------
		# angle between real and estimated rotation
		
		def getAngle(P, Q):
			R = np.dot(P, Q.T)
			cos_theta = (np.trace(R)-1)/2
			cos_theta = max(-1, min(cos_theta, 1)) # clamp result, as it can get slightly outside [-1,1] due to numerical error
			return np.arccos(cos_theta) * (180/np.pi)
		
		# extract estimated rotation		
		# we're doing most of our math (grassmannian, etc) in row-major order, transforms3d seems to default to column-major order
		try :
			T, R, Z, S = transforms3d.affines.decompose(rA.T)
			recovered_rot_matrix = R.T

			#print("initial R matrix:\n{}".format(recovered_rot_matrix)) # converting back from column-major to row-major
			#print("recovered R matrix:\n{}".format(rot_matrix))
			
			angle_difference = getAngle(recovered_rot_matrix,rot_matrix)
			#print("angle difference: {}".format(angle_difference))
		except Exception:
			angle_difference = 180.0
			bad_matrix_count += 1
		matrix_count += 1
		
		'''
		results_recording_dict_2 = {
			"num_matches_list":					[],
			"inlier_frob_norm_list":			[],
			"fullset_frob_norm_list":			[],
			"num_matches_total_possible_list":	[],
			"matrix_frob_list":					[],
			"rotation_difference_angle_list":	[]
		}
		'''
		
		results_recording_dict_2["num_matches_list"].append(num_matches)
		results_recording_dict_2["inlier_frob_norm_list"].append(frob_inliers)
		results_recording_dict_2["fullset_frob_norm_list"].append(frob_fullset)
		results_recording_dict_2["num_matches_total_possible_list"].append(point_set_1.shape[0])
		results_recording_dict_2["matrix_frob_list"].append(matrix_frob)
		results_recording_dict_2["rotation_difference_angle_list"].append(angle_difference)
		
		#assert(0)
		
		#--------------------------------------------------------------------------------------------
		
		if not args.skip_vis:
			# visualize (optional) for point set 1 and point set 2, given various forms of disruption
			'''
			# visual lize initial alignment of landmarks
			pcd1 = o3d.geometry.PointCloud()
			pcd1.points = o3d.utility.Vector3dVector(point_set_1+np.asarray([0.1,0.1,0.1]))			# including a small offset for visibility
			pcd1.paint_uniform_color([1.0,0.0,0.0])
			#coord1 = o3d.geometry.TriangleMesh().create_coordinate_frame(size=10.0,origin=gt_data_dict["kitti_xyz"][frame_id1])
			pcd2 = o3d.geometry.PointCloud()
			pcd2.points = o3d.utility.Vector3dVector(point_set_2)
			pcd2.paint_uniform_color([0.0,0.0,1.0])
			#coord2 = o3d.geometry.TriangleMesh().create_coordinate_frame(size=10.0,origin=gt_data_dict["kitti_xyz"][frame_id2])
			o3d.visualization.draw_geometries([pcd1,pcd2])
			#assert(0)
			'''
			'''
			# visualize grassmanian-generated alignment of landmarks
			pcd1 = o3d.geometry.PointCloud()
			pcd1.points = o3d.utility.Vector3dVector(point_set_1_aligned+np.asarray([0.1,0.1,0.1]))			# including a small offset for visibility
			pcd1.paint_uniform_color([1.0,0.0,0.0])
			#coord1 = o3d.geometry.TriangleMesh().create_coordinate_frame(size=10.0,origin=gt_data_dict["kitti_xyz"][frame_id1])
			pcd2 = o3d.geometry.PointCloud()
			pcd2.points = o3d.utility.Vector3dVector(point_set_2)
			pcd2.paint_uniform_color([0.0,0.0,1.0])
			#coord2 = o3d.geometry.TriangleMesh().create_coordinate_frame(size=10.0,origin=gt_data_dict["kitti_xyz"][frame_id2])
			o3d.visualization.draw_geometries([pcd1,pcd2])
			'''
			
			#TODO
			assert 0, "if you're seeing this, you probably meant to turn off visualization, or restore the commented code above and comment out the code below"
			
			#----------------------------
			# visualize (optional) for point set 1 and point set 2, the grassmannian recovery
			
			#draw lines between corr y coordinates and corrx*rA (corrX_aligned) coordinates to show beleived correlations
			#corrX_aligned, corrY
			
			#TODO
			testing_offset = 0.1
			point_set_1_aligned = point_set_1_aligned + np.asarray([testing_offset,testing_offset,testing_offset])
			corrX_aligned = corrX_aligned + np.asarray([testing_offset,testing_offset,testing_offset])
			scan_points1_aligned = scan_points1_aligned + np.asarray([testing_offset,testing_offset,testing_offset])
			
			#np.set_printoptions(suppress=True)
			#print(corrX_aligned)
			#print(corrY)
			
			# setting up colors
			#color_landmarks1 = [1.0,0.0,0.0]	#red
			#color_landmarks2 = [0.0,0.0,1.0]	#blue
			#color_raw_scan1 = [0.5,0.0,0.0]		#dark red
			#color_raw_scan2 = [0.0,0.0,0.5]		#dark blue
			color_landmarks1 = np.asarray([225,193,7])/255.0	# yellow		pink
			color_landmarks2 = np.asarray([30,136,229])/255.0	# blue			light green
			color_raw_scan1 = np.asarray([216,27,96])/255.0		# magenta		red
			color_raw_scan2 = np.asarray([0,77,64])/255.0		# dark green	dark green
			#color_landmarks1 = np.asarray([254,181,192])/255.0	# yellow		pink
			#color_landmarks2 = np.asarray([0,145,155])/255.0	# blue			light green
			#color_raw_scan1 = np.asarray([212,48,49])/255.0		# magenta		red
			#color_raw_scan2 = np.asarray([24,76,82])/255.0		# dark green	dark green
			
			'''
			# landmark pointclouds
			pcd1 = o3d.geometry.PointCloud()
			pcd2 = o3d.geometry.PointCloud()
			pcd1.points = o3d.utility.Vector3dVector(point_set_1_aligned)
			pcd2.points = o3d.utility.Vector3dVector(point_set_2)
			pcd1.paint_uniform_color(color_landmarks1)
			pcd2.paint_uniform_color(color_landmarks2)
			
			# raw pseudo scan pointclouds
			pcd3 = o3d.geometry.PointCloud()
			pcd4 = o3d.geometry.PointCloud()
			pcd3.points = o3d.utility.Vector3dVector(scan_points1_aligned)
			pcd4.points = o3d.utility.Vector3dVector(scan_points2)
			pcd3.paint_uniform_color(color_raw_scan1)
			pcd4.paint_uniform_color(color_raw_scan2)
			
			# openGL lineset connecting associated landmarks
			lineset_points = np.concatenate((corrX_aligned, corrY), axis=0)
			lineset_lines = np.asarray([[x,x+corrX_aligned.shape[0]] for x in range(corrX_aligned.shape[0])])
			lineset_colors = [[0, 0, 0] for i in range(lineset_points.shape[0])]
			line_set = o3d.geometry.LineSet()
			line_set.points = o3d.utility.Vector3dVector(lineset_points)
			line_set.lines = o3d.utility.Vector2iVector(lineset_lines)
			line_set.colors = o3d.utility.Vector3dVector(lineset_colors)
			
			# line thickness is not supported by my driver's implementation of openGL
			#vis = o3d.visualization.Visualizer()
			#vis.create_window("Pose Visualizer")
			#vis.get_render_option().line_width = 10.0
			
			def make_sphere_list_from_points(points, radius=1.0, color=np.asarray([0.5,0.5,0.5])):
				sphere_list = []
				for i in range(points.shape[0]):
					temp_sphere = o3d.geometry.TriangleMesh().create_sphere(radius, resolution=20)
					temp_sphere.paint_uniform_color(color)
					temp_sphere.translate(points[i,:])
					sphere_list.append(temp_sphere)
				return sphere_list
			
			# spheres representing all the landmarks
			sphere_list1 = make_sphere_list_from_points(point_set_1_aligned,	radius=0.5, color=color_landmarks1)
			sphere_list2 = make_sphere_list_from_points(point_set_2,			radius=0.5, color=color_landmarks2)
			
			# test objects
			coord = o3d.geometry.TriangleMesh().create_coordinate_frame(size=10.0)
			test_sphere = o3d.geometry.TriangleMesh().create_sphere(radius=1.0, resolution=20)
			test_sphere.paint_uniform_color([0.0,1.0,0.0])
			
			geometry_list = []
			#geometry_list.append(pcd1)
			#geometry_list.append(pcd2)
			geometry_list.append(pcd3)
			geometry_list.append(pcd4)
			geometry_list.append(line_set)
			#geometry_list.append(coord)
			#geometry_list.append(test_sphere)
			geometry_list = geometry_list + sphere_list1 + sphere_list2
			o3d.visualization.draw_geometries(geometry_list)
			#assert 0
			'''
			#----------------------------
			# TODO visualize pointclouds and landmarks in original locations, with lines between them showing associations
			# (then make show the raw pointclouds and how they end up aligned)
			
			# associated landmark pointclouds
			pcd1 = o3d.geometry.PointCloud()
			pcd2 = o3d.geometry.PointCloud()
			pcd1.points = o3d.utility.Vector3dVector(corrX)
			pcd2.points = o3d.utility.Vector3dVector(corrY)
			pcd1.paint_uniform_color(color_landmarks1)
			pcd2.paint_uniform_color(color_landmarks2)
			
			# openGL lineset connecting associated landmarks
			lineset_points = np.concatenate((corrX, corrY), axis=0)
			lineset_lines = np.asarray([[x,x+corrX.shape[0]] for x in range(corrX.shape[0])])
			lineset_colors = [[0, 0, 0] for i in range(lineset_points.shape[0])]
			line_set = o3d.geometry.LineSet()
			line_set.points = o3d.utility.Vector3dVector(lineset_points)
			line_set.lines = o3d.utility.Vector2iVector(lineset_lines)
			line_set.colors = o3d.utility.Vector3dVector(lineset_colors)
			
			# raw pseudo scan pointclouds
			pcd3 = o3d.geometry.PointCloud()
			pcd4 = o3d.geometry.PointCloud()
			pcd3.points = o3d.utility.Vector3dVector(scan_points1)
			pcd4.points = o3d.utility.Vector3dVector(scan_points2)
			pcd3.paint_uniform_color(color_raw_scan1)
			pcd4.paint_uniform_color(color_raw_scan2)
			
			geometry_list = []
			geometry_list.append(pcd1)
			geometry_list.append(pcd2)
			geometry_list.append(pcd3)
			geometry_list.append(pcd4)
			geometry_list.append(line_set)
			#geometry_list.append(coord)
			#geometry_list.append(test_sphere)
			#geometry_list = geometry_list + sphere_list1 + sphere_list2
			o3d.visualization.draw_geometries(geometry_list)
			
	elif mode == "test_real_and_synthetic_landmark_association_display":
		#TODO can use temp_loop_idx to keep track of waht loop we're on
		
		# get clusters (one set)
		dso_frame_num1 = list_of_scan_indexes[i]				# list of scan indexes is not nesisarily sequential. "i" is the position in this list, not the specific index of a keyframe from DSO
		frame_id1 = out_dict["frame_id"][dso_frame_num1]
		scan_points1 = out_dict["scan_points"][dso_frame_num1]
		#frame_xyz1 = out_dict["frame_pose"][dso_frame_num1]		# according to DSO
		w2c_matrix1 = out_dict["w2c_matrix"][dso_frame_num1]
		c2w_kitti1 = gt_data_dict["kitti_c2w"][frame_id1]
		
		#dso_frame_num2 = list_of_scan_indexes2[i]
		#frame_id2 = out_dict["frame_id"][dso_frame_num2]
		#scan_points2 = out_dict["scan_points"][dso_frame_num2]
		##frame_xyz2 = out_dict["frame_pose"][dso_frame_num2]		# according to DSO
		#w2c_matrix2 = out_dict["w2c_matrix"][dso_frame_num2]
		#c2w_kitti2 = gt_data_dict["kitti_c2w"][frame_id2]
		
		#--------------------------------------------------------------------------------------------
		# only one set of clusters is needed, as the other will be generated from it in a controlled way for testing
		
		
		# selection of clustering method
		#method_string = "cluster"
		#which_clustering_override = None
		#which_clustering_override = "birch"
		#which_clustering_override = "dbscan"
		#which_clustering_override = args.which_clustering_override
		#use_semantic_labels = False
		
		#if use_semantic_labels:
		#	scan_labels1 = out_dict["scan_labels"][dso_frame_num1]
		##	scan_labels2 = out_dict["scan_labels"][dso_frame_num2]
		scan_labels1 = out_dict["scan_labels"][dso_frame_num1]
		
		# get clusters
		if use_semantic_labels:
			cluster_dict = do_clustering({"scan_points": scan_points1,"scan_labels":scan_labels1,"target_num":TARGET_NUM, "which_clustering_override":which_clustering_override}, method="cluster", remove_noise=True, eng=eng)
		else:
			cluster_dict = do_clustering({"scan_points": scan_points1,"target_num":TARGET_NUM, "which_clustering_override":which_clustering_override}, method="cluster", remove_noise=True, eng=eng)
		cluster_points1 = cluster_dict["cluster_points"]
		
		'''
		if use_semantic_labels:
			cluster_dict = do_clustering({"scan_points": scan_points2,"scan_labels":scan_labels2,"target_num":TARGET_NUM, "which_clustering_override":which_clustering_override}, method="cluster", remove_noise=True, eng=eng)
		else:
			cluster_dict = do_clustering({"scan_points": scan_points2,"target_num":TARGET_NUM, "which_clustering_override":which_clustering_override}, method="cluster", remove_noise=True, eng=eng)
		cluster_points2 = cluster_dict["cluster_points"]
		'''
		
		'''
		# transfer from keyframe coords to global coord using gt (simulating grass alignment)
		scanpts1_h = np.concatenate((cluster_points1, np.ones((cluster_points1.shape[0],1))), axis=1)
		scanpts1_cam_h = np.matmul(w2c_matrix1,np.transpose(scanpts1_h))
		scanpts1_global_h = np.matmul(c2w_kitti1,scanpts1_cam_h)
		scanpts1_global = np.transpose(scanpts1_global_h)[:,0:3]
		
		scanpts2_h = np.concatenate((cluster_points2, np.ones((cluster_points2.shape[0],1))), axis=1)
		scanpts2_cam_h = np.matmul(w2c_matrix2,np.transpose(scanpts2_h))
		scanpts2_global_h = np.matmul(c2w_kitti2,scanpts2_cam_h)
		scanpts2_global = np.transpose(scanpts2_global_h)[:,0:3]
		'''
		
		# random rotation is about the origin, so points need to be moved there
		# other operations only add offsets or are within the existing bounds of the point set
		
		# trying to center the points, poorly
		#cluster_points1 = cluster_points1 - (np.amin(cluster_points1, axis=0)+np.amax(cluster_points1, axis=0))/2.0
		
		mean_coord = np.mean(cluster_points1, axis=0)
		scan_points1 = scan_points1 - mean_coord
		cluster_points1 = cluster_points1 - mean_coord
		#print("mean_coord: {}".format(mean_coord))
		
		#--------------------------------------------------------------------------------------------
		
		# needs to use cluster points from a clustering technique, not raw scan points
		# regular birch is probably best clustering technique, though perhaps multiple should be tested (slightly different distribution of clusters)
		
		point_set_1 = cluster_points1
		point_set_2 = cluster_points1
		
		rng = np.random.default_rng(5)
		
		# generate corresponding set with appropriate outliers and noise
		if args.gen_pcent_ouliers is None:
			assert 0, "gen_pcent_ouliers cannot be None in this mode"
		if args.std_dev_noise is None:
			assert 0, "std_dev_noise cannot be None in this mode"
		
		#TODO set up command line arg control for these
		do_rotation = True
		#do_rotation = False
		do_translation = True
		#do_translation = False
		
		# these default to tru and are controlled through commandline args
		# setting the options to zero on the commandline nullifies thier effect
		do_outliers = True
		do_noise = True
		
		'''
		# rotation
		if do_rotation:
			point_set_2 = uniform_random_rotation_helper(point_set_2, rng)

		# translation
		max_displacement = 45
		v = rng.random((1,3))
		random_offset = (v / np.linalg.norm(v))*rng.random()*max_displacement
		if do_translation:
			point_set_2 = point_set_2 + random_offset
		'''
		
		#create rotation and translation matrices (row-major order)
		#-----------------------------------------
		# https://towardsdatascience.com/the-one-stop-guide-for-transformation-matrices-cea8f609bdb1
		# https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/geometry/row-major-vs-column-major-vector.html

		# rotation matrix
		rot_matrix = uniform_random_rotation_matrix_helper(rng)
		rot_t = np.concatenate((
		  np.concatenate((rot_matrix,np.asarray([
			[0],
			[0],
			[0]
		  ])), axis=1),
		  np.asarray([[0,0,0,1]])
		  ), axis=0
		)
		#print("rot_t:\n{}".format(rot_t))

		# translation matrix
		trans_vector = rng.uniform(low=[-45.0, -45.0, 0.0], high=[45.0, 45.0, 8.0], size=(1,3)) #TODO should be 45,8,45?
		trans_t = np.identity(4)
		trans_t[3,0:3] = trans_vector
		#print("trans_t:\n{}".format(trans_t))
		#-----------------------------------------
		if not do_rotation:
			rot_t = np.identity(4)
		if not do_translation:
			trans_t = np.identity(4)
		
		bdov = 40 #meters, base display offset vector
		
		display_offset_vectors = np.asarray([
			[0,bdov*(-1),0],
			[0,0,bdov*(-1.5)],
			[bdov*(1.5),0,0],
			[bdov,0,0],
			[0,0,bdov*(-1)],
			[bdov*(-2),0,0],
			[0,0,bdov],
			[0,0,bdov*(-2)],
			[0,0,bdov*(-1.5)],
			[bdov*(-1.5),0,0]
		])
		
		#TODO if we're doing really dumb hackery for display purposes:
		if (True):
			print("temp_loop_idx: {}".format(temp_loop_idx))
			
			#TODO modify transformations appropriately
			rot_t = np.identity(4)
			trans_vector = display_offset_vectors[temp_loop_idx,:]
			trans_t = np.identity(4)
			trans_t[3,0:3] = trans_vector
			
			
		
		#compose matrices, transform point_set_2 to get transformed point_set_2
		composed_transform_matrix = np.dot(rot_t,trans_t)
		
		points_temp = np.concatenate((point_set_2,np.ones((point_set_2.shape[0],1))), axis=1)
		points_temp = np.dot(points_temp,composed_transform_matrix)
		point_set_2 = points_temp[:,0:3]
		
		# conceptually:
		#points_rot = np.dot(point_set_2,rot_t,trans_t) # take points, rotate them, then translate
		#-----------------------------------------
		
		# create a clean set of points 2 without noise or outliers for use when checking alignment
		clean_point_set_2 = np.copy(point_set_2)
		
		#TODO DONT USE THIS, USE THE BLOCK BELOW
		'''
		# randomly replace with outliers
		old_count = point_set_2.shape[0]
		remove_num = math.floor(old_count*args.gen_pcent_ouliers)
		lower_xyz_bound = np.amin(point_set_2, axis=0)
		upper_xyz_bound = np.amax(point_set_2, axis=0)
		del_sel = np.sort(rng.choice(old_count,size=remove_num,replace=False))
		remaining = np.delete(point_set_2, del_sel, axis=0)
		new_points = rng.uniform(low=lower_xyz_bound, high=upper_xyz_bound, size=(remove_num,3))
		if do_outliers:
			point_set_2 = np.concatenate((remaining,new_points), axis=0)
		'''
		
		# randomly replace with outliers, preserve order
		old_count = point_set_2.shape[0]
		remove_num = math.floor(old_count*args.gen_pcent_ouliers)
		#print("num outliers to generate: {}".format(remove_num))
		rand_mask = np.full(old_count, True)
		rand_mask[:remove_num] = False
		rng.shuffle(rand_mask)
		lower_xyz_bound = np.amin(point_set_2, axis=0)
		upper_xyz_bound = np.amax(point_set_2, axis=0)
		all_outliers = rng.uniform(low=lower_xyz_bound, high=upper_xyz_bound, size=(old_count,3))
		if do_outliers:
			point_set_2 = np.where(rand_mask, point_set_2.transpose(), all_outliers.transpose()).transpose()
		
		#TODO DONT USE THIS, USE THE BLOCK BELOW
		'''
		# randomly replace with outliers, outside initial shape
		#TODO alternative approach to outliers, placing them outside the pointclouds
		old_count = point_set_2.shape[0]
		remove_num = math.floor(old_count*args.gen_pcent_ouliers)
		#print("num outliers to generate: {}".format(remove_num))
		lower_xyz_bound = np.amin(point_set_2, axis=0)
		upper_xyz_bound = np.amax(point_set_2, axis=0)
		del_sel = np.sort(rng.choice(old_count,size=remove_num,replace=False))
		remaining = np.delete(point_set_2, del_sel, axis=0)
		#new_points = rng.uniform(low=lower_xyz_bound, high=upper_xyz_bound, size=(remove_num,3))
		diagonal_dist = np.linalg.norm(upper_xyz_bound-lower_xyz_bound)
		radius = 0.5*diagonal_dist
		unit_vecs = np.full((remove_num,3), fill_value=[1.0,0.0,0.0])
		#print("unit_vecs.shape: {}".format(unit_vecs.shape))
		if unit_vecs.shape[0] > 0:
			rotated_vecs = np.asarray([uniform_random_rotation_helper(vec, rng)[0] for vec in unit_vecs])
		else:
			# if there aren't any vectors, don't bother trying to rotate them
			rotated_vecs = unit_vecs
		#print("rotated_vecs.shape: {}".format(rotated_vecs.shape))
		#offset_vecs = rotated_vecs*rng.normal(loc=0.0, scale=noise_std_dev, size=(rotated_vecs.shape[0],1))
		#offset_vecs = rotated_vecs*scipy.stats.halfnorm.rvs(scale=noise_std_dev, size=(rotated_vecs.shape[0],1))
		new_points = rotated_vecs*(3*radius) # produce a bunch of outliers somewhere outside the pointclouds being aligned
		#print("new_points.shape: {}".format(new_points.shape))
		if do_outliers:
			point_set_2 = np.concatenate((remaining,new_points), axis=0)
		'''
		
		#TODO #TODO #TODO #TODO #TODO #TODO #TODO #TODO #TODO #TODO #TODO
		'''
		# this is for "gentler outlier" remember to remove this later and reinstate the correct paragraph above
		# randomly replace with outliers, outside initial shape, preserve order
		old_count = point_set_2.shape[0]
		remove_num = math.floor(old_count*args.gen_pcent_ouliers)
		#print("num outliers to generate: {}".format(remove_num))
		rand_mask = np.full(old_count, True)
		rand_mask[:remove_num] = False
		rng.shuffle(rand_mask)
		lower_xyz_bound = np.amin(point_set_2, axis=0)
		upper_xyz_bound = np.amax(point_set_2, axis=0)
		diagonal_dist = np.linalg.norm(upper_xyz_bound-lower_xyz_bound)
		radius = 0.5*diagonal_dist
		unit_vecs = np.full((old_count,3), fill_value=[1.0,0.0,0.0])
		if unit_vecs.shape[0] > 0:
			rotated_vecs = np.asarray([uniform_random_rotation_helper(vec, rng)[0] for vec in unit_vecs])
		else:
			rotated_vecs = unit_vecs
		all_outliers = rotated_vecs*(3*radius) # produce a bunch of outliers somewhere outside the shape being aligned
		if do_outliers:
			point_set_2 = np.where(rand_mask, point_set_2.transpose(), all_outliers.transpose()).transpose()
		assert(args.gen_pcent_ouliers > 0.0)
		'''
		#TODO unused
		# add random noise (multiplicative, gaussian)
		#mul_noise = rng.normal(loc=1.0, scale=noise_std_dev, size=point_set_2.shape)
		#point_set_2 = np.multiply(point_set_2, mul_noise)
		
		#TODO DONT USE THIS, USE THE BLOCK BELOW -------> is there actually any difference? I don't think so be we replaced it at the same time as the order-preserving outlier code blocks above
		'''
		# add random noise (offset, gaussian)
		unit_vecs = np.full((point_set_2.shape[0],3), fill_value=[1.0,0.0,0.0])
		rotated_vecs = np.asarray([uniform_random_rotation_helper(vec, rng)[0] for vec in unit_vecs])
		#offset_vecs = rotated_vecs*rng.normal(loc=0.0, scale=args.std_dev_noise, size=(rotated_vecs.shape[0],1))
		offset_vecs = rotated_vecs*scipy.stats.halfnorm.rvs(scale=args.std_dev_noise, size=(rotated_vecs.shape[0],1))
		if do_noise:
			point_set_2 = point_set_2 + offset_vecs
		'''
		
		# add random noise (offset, gaussian)
		unit_vecs = np.full((point_set_2.shape[0],3), fill_value=[1.0,0.0,0.0])
		rotated_vecs = np.asarray([uniform_random_rotation_helper(vec, rng)[0] for vec in unit_vecs])
		#offset_vecs = rotated_vecs*rng.normal(loc=0.0, scale=args.std_dev_noise, size=(rotated_vecs.shape[0],1))
		offset_vecs = rotated_vecs*scipy.stats.halfnorm.rvs(scale=args.std_dev_noise, size=(rotated_vecs.shape[0],1))
		if do_noise:
			point_set_2 = point_set_2 + offset_vecs
		
		point_set_2 = np.ascontiguousarray(point_set_2)
		#--------------------------------------------------------------------------------------------
		
		# run grassmannian, measure accuracy of transform and number of associations found
		out_dict2 = eng.grassgraph_assoc_test({"X":matlab.double(point_set_1),"Y":matlab.double(point_set_2),"p":grass_parameters})
		
		rA = np.asarray(out_dict2["rA"])
		num_matches = out_dict2["num_matches"]
		#print("rA matrix:\n{}".format(rA))
		print("num_matches: {} out of {}".format(num_matches, point_set_1.shape[0]))
		
		'''
		# Y = X*rA
		# pts2 = pts1*rA
		'''
		
		# transform point_set_1 to match point_set_2
		point_set_1_h = np.concatenate((point_set_1,np.ones((point_set_1.shape[0],1))), axis=1)
		point_set_1_h_aligned = np.matmul(point_set_1_h,rA)
		point_set_1_aligned = point_set_1_h_aligned[:,0:3]
		
		corrX = np.asarray(out_dict2["corrX"])
		corrY = np.asarray(out_dict2["corrY"])
		
		#XrA = np.asarray(out_dict2["XrA"]) 	# XrA and corrX_aligned are the same thing
		
		# transform corr.X to match corr.Y
		corrX_h = np.concatenate((corrX,np.ones((corrX.shape[0],1))), axis=1)
		corrX_h_aligned = np.matmul(corrX_h,rA)
		corrX_aligned = corrX_h_aligned[:,0:3]
		
		#print("corrX.shape: {}".format(corrX.shape))
		#print("corrY.shape: {}".format(corrY.shape))
		#print("XrA.shape: {}".format(XrA.shape))
		
		#XrA = corr.X*rA
		#temp1 = XrA-corrX			#-> not alligned well
		#temp2 = XrA-corrX_aligned	#-> zero
		#print("diff:\n{}".format(temp1))
		#print("diff:\n{}".format(temp2))
		
		# transform scan_points1 using A
		scan_points1_h = np.concatenate((scan_points1,np.ones((scan_points1.shape[0],1))), axis=1)
		scan_points2_h = np.matmul(scan_points1_h,composed_transform_matrix)
		scan_points2 = scan_points2_h[:,0:3]
		
		# transform scan_points1 using rA
		scan_points1_h = np.concatenate((scan_points1,np.ones((scan_points1.shape[0],1))), axis=1)
		scan_points1_h_aligned = np.matmul(scan_points1_h,rA)
		scan_points1_aligned = scan_points1_h_aligned[:,0:3]
		
		#--------------------------------------------------------------------------------------------
		# run other assoc methods
		# TODO
		
		
		#--------------------------------------------------------------------------------------------
		# collect measures and write out result line (for these outlier and noise settings) for later summarization
		
		# num outliers
		# closeness of inlier points
			# transfer corr.x and corr.y
		# closeness of all points, from transform
		
		# when checking alignment quality, use a copy of the point set from before outliers and noise are added (clean_point_set_2)
		# set 1 or corr.X are always the ones transformed to match set 2 or corr.Y
		# set 2 is the one that has noise and outliers applied, so set 1 should match the clean copy of set 2 perfectly when transformed correctly
		#print("")
		#print("point_set_1_aligned.shape: {}".format(point_set_1_aligned.shape))
		#print("clean_point_set_2.shape: {}".format(clean_point_set_2.shape))
		#print("corrY.shape: {}".format(corrY.shape))
		#print("corrX_aligned.shape: {}".format(corrX_aligned.shape))
		
		#point_dists_inliers = np.linalg.norm(clean_point_set_2-point_set_1_aligned,axis=1)
		#point_dists_fullset = np.linalg.norm(corrY-corrX_aligned,axis=1)
		#print("point_dists_matches.shape: {}".format(point_dists_matches.shape))
		#print("point_dists_fullset.shape: {}".format(point_dists_fullset.shape))
		
		#print("point_dists_inliers.shape: {}".format(point_dists_inliers.shape))
		#print("point_dists_fullset.shape: {}".format(point_dists_fullset.shape))
		#print("(clean_point_set_2-point_set_1_aligned).shape: {}".format((clean_point_set_2-point_set_1_aligned).shape))
		#print("(corrY-corrX_aligned).shape: {}".format((corrY-corrX_aligned).shape))
		
		#TODO don't use, very wrong, only preserved for posterity
		frob_inliers = np.linalg.norm((clean_point_set_2-point_set_1_aligned),'fro')
		frob_fullset = np.linalg.norm((corrY-corrX_aligned),'fro')
		#--------------------------------------------------------------------------------------------
		# frob between recovered and initial transformation matrix
		
		matrix_frob = np.linalg.norm((rA-composed_transform_matrix),'fro')
		#print("frob_norm: {}".format(matrix_frob))
		
		#--------------------------------------------------------------------------------------------
		# angle between real and estimated rotation
		
		def getAngle(P, Q):
			R = np.dot(P, Q.T)
			cos_theta = (np.trace(R)-1)/2
			cos_theta = max(-1, min(cos_theta, 1)) # clamp result, as it can get slightly outside [-1,1] due to numerical error
			return np.arccos(cos_theta) * (180/np.pi)
		
		# extract estimated rotation		
		# we're doing most of our math (grassmannian, etc) in row-major order, transforms3d seems to default to column-major order
		try :
			T, R, Z, S = transforms3d.affines.decompose(rA.T)
			recovered_rot_matrix = R.T

			#print("initial R matrix:\n{}".format(recovered_rot_matrix)) # converting back from column-major to row-major
			#print("recovered R matrix:\n{}".format(rot_matrix))
			
			angle_difference = getAngle(recovered_rot_matrix,rot_matrix)
			#print("angle difference: {}".format(angle_difference))
		except Exception:
			angle_difference = 180.0
			bad_matrix_count += 1
		matrix_count += 1
		
		'''
		results_recording_dict_2 = {
			"num_matches_list":					[],
			"inlier_frob_norm_list":			[],
			"fullset_frob_norm_list":			[],
			"num_matches_total_possible_list":	[],
			"matrix_frob_list":					[],
			"rotation_difference_angle_list":	[]
		}
		'''
		
		results_recording_dict_2["num_matches_list"].append(num_matches)
		results_recording_dict_2["inlier_frob_norm_list"].append(frob_inliers)
		results_recording_dict_2["fullset_frob_norm_list"].append(frob_fullset)
		results_recording_dict_2["num_matches_total_possible_list"].append(point_set_1.shape[0])
		results_recording_dict_2["matrix_frob_list"].append(matrix_frob)
		results_recording_dict_2["rotation_difference_angle_list"].append(angle_difference)
		
		#assert(0)
		
		#--------------------------------------------------------------------------------------------
		
		if not args.skip_vis:
			# visualize (optional) for point set 1 and point set 2, given various forms of disruption
			'''
			# visual lize initial alignment of landmarks
			pcd1 = o3d.geometry.PointCloud()
			pcd1.points = o3d.utility.Vector3dVector(point_set_1+np.asarray([0.1,0.1,0.1]))			# including a small offset for visibility
			pcd1.paint_uniform_color([1.0,0.0,0.0])
			#coord1 = o3d.geometry.TriangleMesh().create_coordinate_frame(size=10.0,origin=gt_data_dict["kitti_xyz"][frame_id1])
			pcd2 = o3d.geometry.PointCloud()
			pcd2.points = o3d.utility.Vector3dVector(point_set_2)
			pcd2.paint_uniform_color([0.0,0.0,1.0])
			#coord2 = o3d.geometry.TriangleMesh().create_coordinate_frame(size=10.0,origin=gt_data_dict["kitti_xyz"][frame_id2])
			o3d.visualization.draw_geometries([pcd1,pcd2])
			#assert(0)
			'''
			'''
			# visualize grassmanian-generated alignment of landmarks
			pcd1 = o3d.geometry.PointCloud()
			pcd1.points = o3d.utility.Vector3dVector(point_set_1_aligned+np.asarray([0.1,0.1,0.1]))			# including a small offset for visibility
			pcd1.paint_uniform_color([1.0,0.0,0.0])
			#coord1 = o3d.geometry.TriangleMesh().create_coordinate_frame(size=10.0,origin=gt_data_dict["kitti_xyz"][frame_id1])
			pcd2 = o3d.geometry.PointCloud()
			pcd2.points = o3d.utility.Vector3dVector(point_set_2)
			pcd2.paint_uniform_color([0.0,0.0,1.0])
			#coord2 = o3d.geometry.TriangleMesh().create_coordinate_frame(size=10.0,origin=gt_data_dict["kitti_xyz"][frame_id2])
			o3d.visualization.draw_geometries([pcd1,pcd2])
			'''
			
			
			#----------------------------
			# visualize (optional) for point set 1 and point set 2, the grassmannian recovery
			
			#draw lines between corr y coordinates and corrx*rA (corrX_aligned) coordinates to show beleived correlations
			#corrX_aligned, corrY
			
			#TODO
			testing_offset = 0.1
			point_set_1_aligned = point_set_1_aligned + np.asarray([testing_offset,testing_offset,testing_offset])
			corrX_aligned = corrX_aligned + np.asarray([testing_offset,testing_offset,testing_offset])
			scan_points1_aligned = scan_points1_aligned + np.asarray([testing_offset,testing_offset,testing_offset])
			
			#np.set_printoptions(suppress=True)
			#print(corrX_aligned)
			#print(corrY)
			
			# basic colors
			#color_landmarks1 = [1.0,0.0,0.0]	#red
			#color_landmarks2 = [0.0,0.0,1.0]	#blue
			#color_raw_scan1 = [0.5,0.0,0.0]		#dark red
			#color_raw_scan2 = [0.0,0.0,0.5]		#dark blue
			
			# colorblond friendly colors
			#color_landmarks1 = np.asarray([225,193,7])/255.0	# yellow		pink
			#color_landmarks2 = np.asarray([30,136,229])/255.0	# blue			light green
			#color_raw_scan1 = np.asarray([216,27,96])/255.0		# magenta		red
			#color_raw_scan2 = np.asarray([0,77,64])/255.0		# dark green	dark green
			
			# what the colorblind friendly colors look like to someone colorblind
			#color_landmarks1 = np.asarray([254,181,192])/255.0	# yellow		pink
			#color_landmarks2 = np.asarray([0,145,155])/255.0	# blue			light green
			#color_raw_scan1 = np.asarray([212,48,49])/255.0		# magenta		red
			#color_raw_scan2 = np.asarray([24,76,82])/255.0		# dark green	dark green
			
			# blue and red
			color_landmarks1 = np.asarray([0,0,200])/255.0	#dark blue
			color_landmarks2 = np.asarray([200,0,0])/255.0	#red
			color_raw_scan1 = np.asarray([100,100,255])/255.0	#light blue
			color_raw_scan2 = np.asarray([255,100,100])/255.0	#pink
			
			doing_semantic_coloring = True #TODO
			
			# for semantic coloring, make landmarks more visually distinctive
			if(doing_semantic_coloring):
				color_landmarks1 = np.asarray([255,0,255])/255.0	#dark blue
				color_landmarks2 = np.asarray([255,0,255])/255.0	#red
			
			# blue and red (flipped)
			#color_landmarks1 = np.asarray([127,127,255])/255.0	#blue
			#color_landmarks2 = np.asarray([255,127,127])/255.0	#pink
			#color_raw_scan1 = np.asarray([0,0,255])/255.0	#blue
			#color_raw_scan2 = np.asarray([255,0,0])/255.0	#red
			
			# blue and red
			#color_landmarks1 = np.asarray([127,127,150])/255.0	#dark blue
			#color_landmarks2 = np.asarray([150,127,127])/255.0	#red
			#color_raw_scan1 = np.asarray([0,0,255])/255.0	#light blue
			#color_raw_scan2 = np.asarray([255,0,0])/255.0	#pink
			
			def make_sphere_list_from_points(points, radius=1.0, color=np.asarray([0.5,0.5,0.5])):
				sphere_list = []
				for i in range(points.shape[0]):
					temp_sphere = o3d.geometry.TriangleMesh().create_sphere(radius, resolution=20)
					temp_sphere.paint_uniform_color(color)
					temp_sphere.translate(points[i,:])
					sphere_list.append(temp_sphere)
				return sphere_list
			
			
			
			# landmark pointclouds
			pcd1 = o3d.geometry.PointCloud()
			pcd2 = o3d.geometry.PointCloud()
			pcd1.points = o3d.utility.Vector3dVector(point_set_1_aligned)
			pcd2.points = o3d.utility.Vector3dVector(point_set_2)
			pcd1.paint_uniform_color(color_landmarks1)
			pcd2.paint_uniform_color(color_landmarks2)
			
			#TODO blue and red
			#color_raw_scan1 = np.asarray([0,0,255])/255.0	#light blue
			#color_raw_scan2 = np.asarray([255,0,0])/255.0	#pink
			
			# raw pseudo scan pointclouds
			pcd3 = o3d.geometry.PointCloud()
			pcd4 = o3d.geometry.PointCloud()
			pcd3.points = o3d.utility.Vector3dVector(scan_points1_aligned)
			pcd4.points = o3d.utility.Vector3dVector(scan_points2)
			pcd3.paint_uniform_color(color_raw_scan1)
			pcd4.paint_uniform_color(color_raw_scan2)
			
			# openGL lineset connecting associated landmarks
			lineset_points = np.concatenate((corrX_aligned, corrY), axis=0)
			lineset_lines = np.asarray([[x,x+corrX_aligned.shape[0]] for x in range(corrX_aligned.shape[0])])
			lineset_colors = [[0, 0, 0] for i in range(lineset_points.shape[0])]
			line_set = o3d.geometry.LineSet()
			line_set.points = o3d.utility.Vector3dVector(lineset_points)
			line_set.lines = o3d.utility.Vector2iVector(lineset_lines)
			line_set.colors = o3d.utility.Vector3dVector(lineset_colors)
			
			# line thickness is not supported by my driver's implementation of openGL
			#vis = o3d.visualization.Visualizer()
			#vis.create_window("Pose Visualizer")
			#vis.get_render_option().line_width = 10.0
			
			# spheres representing all the landmarks
			sphere_list1 = make_sphere_list_from_points(point_set_1_aligned,	radius=0.5, color=color_landmarks1)
			sphere_list2 = make_sphere_list_from_points(point_set_2,			radius=0.5, color=color_landmarks2)
			
			# test objects
			coord = o3d.geometry.TriangleMesh().create_coordinate_frame(size=10.0)
			test_sphere = o3d.geometry.TriangleMesh().create_sphere(radius=1.0, resolution=20)
			test_sphere.paint_uniform_color([0.0,1.0,0.0])
			
			geometry_list = []
			#geometry_list.append(pcd1)									# aligned landmarks using grassmannian solution
			#geometry_list.append(pcd2)
			geometry_list.append(pcd3)									# aligned plointclouds using grassmannian solution
			geometry_list.append(pcd4)
			#geometry_list.append(line_set)								# openGL lineset connecting associated landmarks
			#geometry_list.append(coord)								# coordinate frame
			#geometry_list.append(test_sphere)							# test sphere
			#geometry_list = geometry_list + sphere_list1 + sphere_list2	# spheres representing all the landmarks
			o3d.visualization.draw_geometries(geometry_list)
			#assert 0
			
			#----------------------------
			# TODO visualize pointclouds and landmarks in original locations, with lines between them showing associations
			# (then make show the raw pointclouds and how they end up aligned)
			
			'''
			ROAD_LABEL = 0
			SIDEWALK_LABEL = 1
			BUILDING_LABEL = 2
			VEGETATION_LABEL = 8
			CAR_LABEL = 13
			
			# generate colored labels based on semantics
			#print("scan_labels1.shape: {}".format(scan_labels1.shape))
			scan_label_colors = np.full((scan_labels1.shape[0],3), [0.0 , 0.0, 0.0])
			scan_label_colors[scan_labels1 == ROAD_LABEL]		= np.asarray([80,80,80])/255.0
			scan_label_colors[scan_labels1 == SIDEWALK_LABEL]	= np.asarray([127,127,127])/255.0
			scan_label_colors[scan_labels1 == BUILDING_LABEL]	= np.asarray([127,0,0])/255.0
			scan_label_colors[scan_labels1 == VEGETATION_LABEL]	= np.asarray([0,127,0])/255.0
			scan_label_colors[scan_labels1 == CAR_LABEL]		= np.asarray([0,0,127])/255.0
			# purple -> 255,127,255
			
			# associated landmark pointclouds
			pcd1 = o3d.geometry.PointCloud()
			pcd2 = o3d.geometry.PointCloud()
			pcd1.points = o3d.utility.Vector3dVector(corrX)
			pcd2.points = o3d.utility.Vector3dVector(corrY)
			pcd1.paint_uniform_color(color_landmarks1)
			pcd2.paint_uniform_color(color_landmarks2)
			
			# spheres representing associated landmarks
			sphere_list1 = make_sphere_list_from_points(corrX,	radius=0.5, color=color_landmarks1)
			sphere_list2 = make_sphere_list_from_points(corrY,	radius=0.5, color=color_landmarks2)
			
			# spheres representing all the landmarks
			sphere_list3 = make_sphere_list_from_points(point_set_1,	radius=0.5, color=color_landmarks1)
			sphere_list4 = make_sphere_list_from_points(point_set_2,	radius=0.5, color=color_landmarks2)
			
			# openGL lineset connecting associated landmarks
			lineset_points = np.concatenate((corrX, corrY), axis=0)
			lineset_lines = np.asarray([[x,x+corrX.shape[0]] for x in range(corrX.shape[0])]) # [0,1,2,3,4] -> [0,1,2,3,4, 5,6,7,8,9]
			lineset_colors = [[0, 0, 0] for i in range(lineset_lines.shape[0])]
			
			
			## print("lineset_points.shape: {}".format(lineset_points.shape))
			## print("lineset_lines.shape: {}".format(lineset_lines.shape))
			## print(".shape: {}".format(.shape))
			# check_dist = 5 #TODO not really a reliable criteria for whether an association is correct, except in truly terrible matches
			# check_points_Y = corrY - np.asarray(display_offset_vectors[temp_loop_idx])
			## print("check_points_Y.shape: {}".format(check_points_Y.shape))
			## print("corrX.shape: {}".format(corrX.shape))
			## print("subtraction result: {}".format(corrX - check_points_Y))
			# dists = np.linalg.norm(corrX - check_points_Y, axis=1)
			## print("dists.shape: {}".format(dists.shape))
			# lineset_colors = np.asarray([[0, 0, 0] for i in range(lineset_lines.shape[0])])
			## print("lineset_colors.shape: {}".format(np.asarray(lineset_colors).shape))
			## print("(dists > check_dist).shape: {}".format((dists > check_dist).shape))
			## print("(dists <= check_dist).shape: {}".format((dists <= check_dist).shape))
			# mask = dists > check_dist
			## print("mask.shape: {}".format(mask.shape))
			## print("mask.dtype: {}".format(mask.dtype))
			# lineset_colors[dists > check_dist] = np.asarray([255,0,0])/255.0
			# lineset_colors[dists <= check_dist] = np.asarray([0,255,0])/255.0
			## for j in range(np.asarray(lineset_colors).shape[0]):
			##	if dists[i] > check_dist:
			##		#print("lineset_colors[i,:].shape: {}".format(lineset_colors[i,:].shape))
			##		lineset_colors[i,:] = np.asarray([255,0,0])/255.0
			##	else:
			##		lineset_colors[i,:] = np.asarray([0,255,0])/255.0
			## assert 0
			
			
			line_set = o3d.geometry.LineSet()
			line_set.points = o3d.utility.Vector3dVector(lineset_points)
			line_set.lines = o3d.utility.Vector2iVector(lineset_lines)
			line_set.colors = o3d.utility.Vector3dVector(lineset_colors)
			
			mesh_line_set = LineMesh(lineset_points, lineset_lines, lineset_colors, radius=0.08)
			mesh_line_set_geoms = mesh_line_set.cylinder_segments
			
			# raw pseudo scan pointclouds
			pcd3 = o3d.geometry.PointCloud()
			pcd4 = o3d.geometry.PointCloud()
			pcd3.points = o3d.utility.Vector3dVector(scan_points1)
			pcd4.points = o3d.utility.Vector3dVector(scan_points2)
			if (not doing_semantic_coloring):
				pcd3.paint_uniform_color(color_raw_scan1)
				pcd4.paint_uniform_color(color_raw_scan2)
			else:
				pcd3.colors = o3d.utility.Vector3dVector(scan_label_colors)
				pcd4.colors = o3d.utility.Vector3dVector(scan_label_colors)
			
			# some test objects
			coord = o3d.geometry.TriangleMesh().create_coordinate_frame(size=10.0)
			test_sphere = o3d.geometry.TriangleMesh().create_sphere(radius=1.0, resolution=20)
			
			geometry_list = []
			#geometry_list.append(pcd1)										# pointclouds of associated landmarks
			#geometry_list.append(pcd2)
			geometry_list.append(pcd3)										# pointclouds of scan points (may be semantically colored)
			#TODO geometry_list.append(pcd4)
			#geometry_list.append(line_set)									# lines between correlated landmarks (openGL)
			#TODO geometry_list = geometry_list + mesh_line_set_geoms				# lines between correlated landmarks (cylinders)
			#geometry_list.append(coord)									# a coordinate frame
			#geometry_list.append(test_sphere)								# test sphere
			#geometry_list = geometry_list + sphere_list1 + sphere_list2	# spheres representing associated landmarks
			#TODO geometry_list = geometry_list + sphere_list3 + sphere_list4		# spheres representing all landmarks
			o3d.visualization.draw_geometries(geometry_list)
			'''
			
		pass
	elif mode == "test_real_and_synthetic_landmark_association_grassgraph_data":
		# get clusters (one set)
		'''
		dso_frame_num1 = list_of_scan_indexes[i]				# list of scan indexes is not nesisarily sequential. "i" is the position in this list, not the specific index of a keyframe from DSO
		frame_id1 = out_dict["frame_id"][dso_frame_num1]
		scan_points1 = out_dict["scan_points"][dso_frame_num1]
		#frame_xyz1 = out_dict["frame_pose"][dso_frame_num1]		# according to DSO
		w2c_matrix1 = out_dict["w2c_matrix"][dso_frame_num1]
		c2w_kitti1 = gt_data_dict["kitti_c2w"][frame_id1]
		'''
		
		# get cluster_points1 from alternative source of data, in this case loaded from a matlab object provided with grasgraph
		
		cluster_points1 = shapes_loaded_from_an_alternate_source[i,:,:]
		
		#print("cluster_points1.shape: {}".format(cluster_points1.shape))
		#assert(0)
		
		#--------------------------------------------------------------------------------------------
		# only one set of clusters is needed, as the other will be generated from it in a controlled way for testing
		
		# random rotation is about the origin, so points need to be moved there
		# other operations only add offsets or are within the existing bounds of the point set
		
		# trying to center the points, poorly
		#cluster_points1 = cluster_points1 - (np.amin(cluster_points1, axis=0)+np.amax(cluster_points1, axis=0))/2.0
		
		mean_coord = np.mean(cluster_points1, axis=0)
		cluster_points1 = cluster_points1 - mean_coord
		#print("mean_coord: {}".format(mean_coord))
		
		#--------------------------------------------------------------------------------------------
		
		# needs to use cluster points from a clustering technique, not raw scan points
		# regular birch is probably best clustering technique, though perhaps multiple should be tested (slightly different distribution of clusters)
		
		point_set_1 = cluster_points1
		point_set_2 = cluster_points1
		
		rng = np.random.default_rng(5)
		
		# generate corresponding set with appropriate outliers and noise
		if args.gen_pcent_ouliers is None:
			assert 0, "gen_pcent_ouliers cannot be None in this mode"
		if args.std_dev_noise is None:
			assert 0, "std_dev_noise cannot be None in this mode"
		
		#TODO set up command line arg control for these
		do_rotation = True
		#do_rotation = False
		do_translation = True
		#do_translation = False
		
		# these default to tru and are controlled through commandline args
		# setting the options to zero on the commandline nullifies thier effect
		do_outliers = True
		do_noise = True
		
		
		#create rotation and translation matrices (row-major order)
		#-----------------------------------------
		# https://towardsdatascience.com/the-one-stop-guide-for-transformation-matrices-cea8f609bdb1
		# https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/geometry/row-major-vs-column-major-vector.html

		# rotation matrix
		rot_matrix = uniform_random_rotation_matrix_helper(rng)
		rot_t = np.concatenate((
		  np.concatenate((rot_matrix,np.asarray([
			[0],
			[0],
			[0]
		  ])), axis=1),
		  np.asarray([[0,0,0,1]])
		  ), axis=0
		)
		#print("rot_t:\n{}".format(rot_t))

		# translation matrix
		trans_vector = rng.uniform(low=[-45.0, -45.0, 0.0], high=[45.0, 45.0, 8.0], size=(1,3)) #TODO should be 45,8,45?
		trans_t = np.identity(4)
		trans_t[3,0:3] = trans_vector
		#print("trans_t:\n{}".format(trans_t))
		#-----------------------------------------
		if not do_rotation:
			rot_t = np.identity(4)
		if not do_translation:
			trans_t = np.identity(4)
		
		#compose matrices, transform point_set_2 to get transformed point_set_2
		composed_transform_matrix = np.dot(rot_t,trans_t)
		
		points_temp = np.concatenate((point_set_2,np.ones((point_set_2.shape[0],1))), axis=1)
		points_temp = np.dot(points_temp,composed_transform_matrix)
		point_set_2 = points_temp[:,0:3]

		# conceptually:
		#points_rot = np.dot(point_set_2,rot_t,trans_t) # take points, rotate them, then translate
		#-----------------------------------------
		
		# create a clean set of points 2 without noise or outliers for use when checking alignment
		clean_point_set_2 = np.copy(point_set_2)
		
		#TODO DONT USE THIS, USE THE BLOCK BELOW
		'''
		# randomly replace with outliers
		old_count = point_set_2.shape[0]
		remove_num = math.floor(old_count*args.gen_pcent_ouliers)
		lower_xyz_bound = np.amin(point_set_2, axis=0)
		upper_xyz_bound = np.amax(point_set_2, axis=0)
		del_sel = np.sort(rng.choice(old_count,size=remove_num,replace=False))
		remaining = np.delete(point_set_2, del_sel, axis=0)
		new_points = rng.uniform(low=lower_xyz_bound, high=upper_xyz_bound, size=(remove_num,3))
		if do_outliers:
			point_set_2 = np.concatenate((remaining,new_points), axis=0)
		'''
		
		# randomly replace with outliers, preserve order
		old_count = point_set_2.shape[0]
		remove_num = math.floor(old_count*args.gen_pcent_ouliers)
		#print("num outliers to generate: {}".format(remove_num))
		rand_mask = np.full(old_count, True)
		rand_mask[:remove_num] = False
		rng.shuffle(rand_mask)
		lower_xyz_bound = np.amin(point_set_2, axis=0)
		upper_xyz_bound = np.amax(point_set_2, axis=0)
		all_outliers = rng.uniform(low=lower_xyz_bound, high=upper_xyz_bound, size=(old_count,3))
		if do_outliers:
			point_set_2 = np.where(rand_mask, point_set_2.transpose(), all_outliers.transpose()).transpose()
		
		#TODO DONT USE THIS, USE THE BLOCK BELOW
		'''
		# randomly replace with outliers, outside initial shape
		#TODO alternative approach to outliers, placing them outside the pointclouds
		old_count = point_set_2.shape[0]
		remove_num = math.floor(old_count*args.gen_pcent_ouliers)
		#print("num outliers to generate: {}".format(remove_num))
		lower_xyz_bound = np.amin(point_set_2, axis=0)
		upper_xyz_bound = np.amax(point_set_2, axis=0)
		del_sel = np.sort(rng.choice(old_count,size=remove_num,replace=False))
		remaining = np.delete(point_set_2, del_sel, axis=0)
		#new_points = rng.uniform(low=lower_xyz_bound, high=upper_xyz_bound, size=(remove_num,3))
		diagonal_dist = np.linalg.norm(upper_xyz_bound-lower_xyz_bound)
		radius = 0.5*diagonal_dist
		unit_vecs = np.full((remove_num,3), fill_value=[1.0,0.0,0.0])
		#print("unit_vecs.shape: {}".format(unit_vecs.shape))
		if unit_vecs.shape[0] > 0:
			rotated_vecs = np.asarray([uniform_random_rotation_helper(vec, rng)[0] for vec in unit_vecs])
		else:
			# if there aren't any vectors, don't bother trying to rotate them
			rotated_vecs = unit_vecs
		#print("rotated_vecs.shape: {}".format(rotated_vecs.shape))
		#offset_vecs = rotated_vecs*rng.normal(loc=0.0, scale=noise_std_dev, size=(rotated_vecs.shape[0],1))
		#offset_vecs = rotated_vecs*scipy.stats.halfnorm.rvs(scale=noise_std_dev, size=(rotated_vecs.shape[0],1))
		new_points = rotated_vecs*(3*radius) # produce a bunch of outliers somewhere outside the pointclouds being aligned
		#print("new_points.shape: {}".format(new_points.shape))
		if do_outliers:
			point_set_2 = np.concatenate((remaining,new_points), axis=0)
		'''
		
		#TODO #TODO #TODO #TODO #TODO #TODO #TODO #TODO #TODO #TODO #TODO
		'''
		# this is for "gentler outlier" remember to remove this later and reinstate the correct paragraph above
		# randomly replace with outliers, outside initial shape, preserve order
		old_count = point_set_2.shape[0]
		remove_num = math.floor(old_count*args.gen_pcent_ouliers)
		#print("num outliers to generate: {}".format(remove_num))
		rand_mask = np.full(old_count, True)
		rand_mask[:remove_num] = False
		rng.shuffle(rand_mask)
		lower_xyz_bound = np.amin(point_set_2, axis=0)
		upper_xyz_bound = np.amax(point_set_2, axis=0)
		diagonal_dist = np.linalg.norm(upper_xyz_bound-lower_xyz_bound)
		radius = 0.5*diagonal_dist
		unit_vecs = np.full((old_count,3), fill_value=[1.0,0.0,0.0])
		if unit_vecs.shape[0] > 0:
			rotated_vecs = np.asarray([uniform_random_rotation_helper(vec, rng)[0] for vec in unit_vecs])
		else:
			rotated_vecs = unit_vecs
		all_outliers = rotated_vecs*(3*radius) # produce a bunch of outliers somewhere outside the shape being aligned
		if do_outliers:
			point_set_2 = np.where(rand_mask, point_set_2.transpose(), all_outliers.transpose()).transpose()
		assert(args.gen_pcent_ouliers > 0.0)
		'''
		#TODO unused
		# add random noise (multiplicative, gaussian)
		#mul_noise = rng.normal(loc=1.0, scale=noise_std_dev, size=point_set_2.shape)
		#point_set_2 = np.multiply(point_set_2, mul_noise)
		
		#TODO DONT USE THIS, USE THE BLOCK BELOW -------> is there actually any difference? I don't think so be we replaced it at the same time as the order-preserving outlier code blocks above
		'''
		# add random noise (offset, gaussian)
		unit_vecs = np.full((point_set_2.shape[0],3), fill_value=[1.0,0.0,0.0])
		rotated_vecs = np.asarray([uniform_random_rotation_helper(vec, rng)[0] for vec in unit_vecs])
		#offset_vecs = rotated_vecs*rng.normal(loc=0.0, scale=args.std_dev_noise, size=(rotated_vecs.shape[0],1))
		offset_vecs = rotated_vecs*scipy.stats.halfnorm.rvs(scale=args.std_dev_noise, size=(rotated_vecs.shape[0],1))
		if do_noise:
			point_set_2 = point_set_2 + offset_vecs
		'''
		
		# add random noise (offset, gaussian)
		unit_vecs = np.full((point_set_2.shape[0],3), fill_value=[1.0,0.0,0.0])
		rotated_vecs = np.asarray([uniform_random_rotation_helper(vec, rng)[0] for vec in unit_vecs])
		#offset_vecs = rotated_vecs*rng.normal(loc=0.0, scale=args.std_dev_noise, size=(rotated_vecs.shape[0],1))
		offset_vecs = rotated_vecs*scipy.stats.halfnorm.rvs(scale=args.std_dev_noise, size=(rotated_vecs.shape[0],1))
		if do_noise:
			point_set_2 = point_set_2 + offset_vecs
		
		point_set_2 = np.ascontiguousarray(point_set_2)
		#--------------------------------------------------------------------------------------------
		
		# run grassmannian, measure accuracy of transform and number of associations found
		out_dict2 = eng.grassgraph_assoc_test({"X":matlab.double(point_set_1),"Y":matlab.double(point_set_2),"p":grass_parameters})
		
		rA = np.asarray(out_dict2["rA"])
		num_matches = out_dict2["num_matches"]
		#print("rA matrix:\n{}".format(rA))
		print("num_matches: {} out of {}".format(num_matches, point_set_1.shape[0]))
		
		'''
		# Y = X*rA
		# pts2 = pts1*rA
		'''
		
		# transform point_set_1 to match point_set_2
		point_set_1_h = np.concatenate((point_set_1,np.ones((point_set_1.shape[0],1))), axis=1)
		point_set_1_h_aligned = np.matmul(point_set_1_h,rA)
		point_set_1_aligned = point_set_1_h_aligned[:,0:3]
		
		corrX = np.asarray(out_dict2["corrX"])
		corrY = np.asarray(out_dict2["corrY"])
		
		#XrA = np.asarray(out_dict2["XrA"]) 	# XrA and corrX_aligned are the same thing
		
		# transform corr.X to match corr.Y
		corrX_h = np.concatenate((corrX,np.ones((corrX.shape[0],1))), axis=1)
		corrX_h_aligned = np.matmul(corrX_h,rA)
		corrX_aligned = corrX_h_aligned[:,0:3]
		
		#print("corrX.shape: {}".format(corrX.shape))
		#print("corrY.shape: {}".format(corrY.shape))
		#print("XrA.shape: {}".format(XrA.shape))
		
		#XrA = corr.X*rA
		#temp1 = XrA-corrX			#-> not alligned well
		#temp2 = XrA-corrX_aligned	#-> zero
		#print("diff:\n{}".format(temp1))
		#print("diff:\n{}".format(temp2))
		
		
		#--------------------------------------------------------------------------------------------
		# run other assoc methods
		# TODO
		
		
		#--------------------------------------------------------------------------------------------
		# collect measures and write out result line (for these outlier and noise settings) for later summarization
		
		# num outliers
		# closeness of inlier points
			# transfer corr.x and corr.y
		# closeness of all points, from transform
		
		# when checking alignment quality, use a copy of the point set from before outliers and noise are added (clean_point_set_2)
		# set 1 or corr.X are always the ones transformed to match set 2 or corr.Y
		# set 2 is the one that has noise and outliers applied, so set 1 should match the clean copy of set 2 perfectly when transformed correctly
		#print("")
		#print("point_set_1_aligned.shape: {}".format(point_set_1_aligned.shape))
		#print("clean_point_set_2.shape: {}".format(clean_point_set_2.shape))
		#print("corrY.shape: {}".format(corrY.shape))
		#print("corrX_aligned.shape: {}".format(corrX_aligned.shape))
		
		#point_dists_inliers = np.linalg.norm(clean_point_set_2-point_set_1_aligned,axis=1)
		#point_dists_fullset = np.linalg.norm(corrY-corrX_aligned,axis=1)
		#print("point_dists_matches.shape: {}".format(point_dists_matches.shape))
		#print("point_dists_fullset.shape: {}".format(point_dists_fullset.shape))
		
		#print("point_dists_inliers.shape: {}".format(point_dists_inliers.shape))
		#print("point_dists_fullset.shape: {}".format(point_dists_fullset.shape))
		#print("(clean_point_set_2-point_set_1_aligned).shape: {}".format((clean_point_set_2-point_set_1_aligned).shape))
		#print("(corrY-corrX_aligned).shape: {}".format((corrY-corrX_aligned).shape))
		
		#TODO don't use, very wrong, only preserved for posterity
		frob_inliers = np.linalg.norm((clean_point_set_2-point_set_1_aligned),'fro')
		frob_fullset = np.linalg.norm((corrY-corrX_aligned),'fro')
		#--------------------------------------------------------------------------------------------
		# frob between recovered and initial transformation matrix
		
		matrix_frob = np.linalg.norm((rA-composed_transform_matrix),'fro')
		#print("frob_norm: {}".format(matrix_frob))
		
		#--------------------------------------------------------------------------------------------
		# angle between real and estimated rotation
		
		def getAngle(P, Q):
			R = np.dot(P, Q.T)
			cos_theta = (np.trace(R)-1)/2
			cos_theta = max(-1, min(cos_theta, 1)) # clamp result, as it can get slightly outside [-1,1] due to numerical error
			return np.arccos(cos_theta) * (180/np.pi)
		
		# extract estimated rotation		
		# we're doing most of our math (grassmannian, etc) in row-major order, transforms3d seems to default to column-major order
		try :
			T, R, Z, S = transforms3d.affines.decompose(rA.T)
			recovered_rot_matrix = R.T

			#print("initial R matrix:\n{}".format(recovered_rot_matrix)) # converting back from column-major to row-major
			#print("recovered R matrix:\n{}".format(rot_matrix))
			
			angle_difference = getAngle(recovered_rot_matrix,rot_matrix)
			#print("angle difference: {}".format(angle_difference))
		except Exception:
			angle_difference = 180.0
			bad_matrix_count += 1
		matrix_count += 1
		
		'''
		results_recording_dict_2 = {
			"num_matches_list":					[],
			"inlier_frob_norm_list":			[],
			"fullset_frob_norm_list":			[],
			"num_matches_total_possible_list":	[],
			"matrix_frob_list":					[],
			"rotation_difference_angle_list":	[]
		}
		'''
		
		results_recording_dict_2["num_matches_list"].append(num_matches)
		results_recording_dict_2["inlier_frob_norm_list"].append(frob_inliers)
		results_recording_dict_2["fullset_frob_norm_list"].append(frob_fullset)
		results_recording_dict_2["num_matches_total_possible_list"].append(point_set_1.shape[0])
		results_recording_dict_2["matrix_frob_list"].append(matrix_frob)
		results_recording_dict_2["rotation_difference_angle_list"].append(angle_difference)
		
		#assert(0)
		
		#--------------------------------------------------------------------------------------------
		
		#print("args.skip_vis: {}".format(args.skip_vis))
		#assert 0
		
		if not args.skip_vis:
			# visualize (optional) for point set 1 and point set 2, given various forms of disruption
			pcd1 = o3d.geometry.PointCloud()
			pcd1.points = o3d.utility.Vector3dVector(point_set_1+np.asarray([0.1,0.1,0.1]))			# including a small offset for visibility
			pcd1.paint_uniform_color([1.0,0.0,0.0])
			#coord1 = o3d.geometry.TriangleMesh().create_coordinate_frame(size=10.0,origin=gt_data_dict["kitti_xyz"][frame_id1])
			pcd2 = o3d.geometry.PointCloud()
			pcd2.points = o3d.utility.Vector3dVector(point_set_2)
			pcd2.paint_uniform_color([0.0,0.0,1.0])
			#coord2 = o3d.geometry.TriangleMesh().create_coordinate_frame(size=10.0,origin=gt_data_dict["kitti_xyz"][frame_id2])
			o3d.visualization.draw_geometries([pcd1,pcd2])
			#assert(0)
			
			# visualize (optional) for point set 1 and point set 2, the grassmannian recovery
			pcd1 = o3d.geometry.PointCloud()
			pcd1.points = o3d.utility.Vector3dVector(point_set_1_aligned+np.asarray([0.1,0.1,0.1]))			# including a small offset for visibility
			pcd1.paint_uniform_color([1.0,0.0,0.0])
			#coord1 = o3d.geometry.TriangleMesh().create_coordinate_frame(size=10.0,origin=gt_data_dict["kitti_xyz"][frame_id1])
			pcd2 = o3d.geometry.PointCloud()
			pcd2.points = o3d.utility.Vector3dVector(point_set_2)
			pcd2.paint_uniform_color([0.0,0.0,1.0])
			#coord2 = o3d.geometry.TriangleMesh().create_coordinate_frame(size=10.0,origin=gt_data_dict["kitti_xyz"][frame_id2])
			o3d.visualization.draw_geometries([pcd1,pcd2])
		
		
		pass



	
	#assert(0)
if (mode == "examine_match_pairs") or (mode == "examine_clusters_points_pairs"):
	vis.destroy_window()
	vis2.destroy_window()

if len(times_taken) > 0:
	print("average time taken: {}".format(np.average(times_taken)))


# dict to record lists of statistics over the sequence
#results_recording_dict = {
#	"frame_1_max":[],			# the maximum limits of points in the frame
#	"frame_1_min":[],
#	"frame_2_max":[],
#	"frame_2_min":[],
#	"num_points_1":[],			# the number of points in the frame. together with the bounds of the bounding box above, rough density can be computed
#	"num_points_2":[],
#	
#	"num_clusters" : [],		# should be the same for both frames, technically for all frames
#	"num_outliers_nn" : [],		# number of cluster points not in a mutual-nn relationship. should be same between sets, see below
#	"dists_nearest_neighbors" : [],	# for those in a nn relationship, how far apart are they?
#	
#	"dists_to_closest" : []		# record all the distance to the nearest point for all points in frame 1, we can apply range gating later, or just plot a distribution
#								# we only record the number of outliers that don't have a pairing within 1m in frame 1, we'll get to the reverse pairing where we consider frame 2 later in the sequence
#}


cluster_output_folder_path = args.cluster_output_folder_path
alignment_output_folder_path = args.alignment_output_folder_path

print("reached stat gen")

if mode == "compare_match_pairs_clusters_alignment_measurement":
	
	#cluster_output_folder_path -> where to save images
	
	print("saving results to: {}".format(cluster_output_folder_path))
	
	if not os.path.exists(cluster_output_folder_path):
	    os.makedirs(cluster_output_folder_path)
	
	logfile_path = "{}/logfile.txt".format(cluster_output_folder_path)
	measure_logfile = open(logfile_path, 'a')
	measure_logfile.write("\n")
	measure_logfile.write("\n")
	measure_logfile.write("\n")
	
	#------------------------------------------------------------------------------------------
	
	# compute (areal) density stats	
	tallest = 0
	heights = []
	#densities = []
	nums_per_area = []
	for i in range(len(results_recording_dict["num_points_1"])):
		frame_1_max = results_recording_dict["frame_1_max"][i]
		frame_1_min = results_recording_dict["frame_1_min"][i]
		#frame_2_max = results_recording_dict["frame_2_max"][i]
		#frame_2_min = results_recording_dict["frame_2_min"][i]
		num_points_1 = results_recording_dict["num_points_1"][i]
		#num_points_2 = results_recording_dict["num_points_2"][i]
		
		dimensions_1 = frame_1_max-frame_1_min
		#volume_1 = np.prod(dimensions_1)
		#dimensions_2 = frame_2_max-frame_2_min		# we don't consider frame 2 as it will be frame 1 elsewhere in the sequence
		#volume_2 = np.prod(dimensions_2)
		
		# tallest scan was 9.511952279192524 meters
		# average was 6.687443940934243
		# the histogram had a sharp fall at around 7.5 meters (about 2.2 stories)
		#volume_1 = np.pi*45*45*7.5
		area_1 = np.pi*45*45
		
		if tallest < dimensions_1[1]:
			tallest = dimensions_1[1]
		heights.append(dimensions_1[1])

		#density = num_points_1/volume_1
		#print("points: {} volume: {} density: {}".format(num_points_1,volume_1,density))
		#densities.append(density)
		num_per_area = num_points_1/area_1
		#print("points: {} area: {} pts per m^2: {}".format(num_points_1,area_1,num_per_area))
		nums_per_area.append(num_per_area)
		
		#assert(0)
		
		pass
	
	# plot heights
	'''
	plt.hist(heights, bins=20)
	plt.title("heights")
	#plt.show()
	print("tallest: {}".format(tallest))
	print("average: {}".format(np.average(np.asarray(heights))))
	'''
	
	# plot volume densities
	'''
	print("min/max/avg density in points per cubic meter: {}/{}/{}".format(
		np.amin(densities),
		np.amax(densities),
		np.average(densities)
	))
	plt.figure()
	plt.hist(densities, bins=20)
	plt.title("densities")
	plt.show()
	'''
	
	# plot areal densities
	'''
	print("min/max/avg points per square meter: {}/{}/{}".format(
		np.amin(nums_per_area),
		np.amax(nums_per_area),
		np.average(nums_per_area)
	))
	plt.figure()
	plt.hist(nums_per_area, bins=20)
	plt.title("Pseudo-Pointcloud Densities in Points Per Area")
	plt.xlabel("Average Densitty in Points per Square Meter")
	plt.ylabel("Pointcloud Count")
	plt.show()
	'''
	
	#------------------------------------------------------------------------------------------
	
	# counts of nn outliers for each frame, counts of outliers where the nearest point is further than some threshold
	# produce histogram of the number of outliers in each frame
	
	'''
	outlier_counts_for_nn_across_frames = results_recording_dict["num_outliers_nn"]
	outlier_counts_at_1m_across_frames = []
	outlier_counts_at_2m5_across_frames = []
	limit_1 = 1
	limit_2 = 2.5
	for i in range(len(results_recording_dict["num_clusters"])):
		num_clusters = results_recording_dict["num_clusters"][i]
		# for each frame:
		
		# take the lists of to-nearest-point dists
		dists_nearest_neighbors = results_recording_dict["dists_nearest_neighbors"][i]
		
		print(dists_nearest_neighbors)
		print(np.asarray(dists_nearest_neighbors) > limit_1)
		print(np.asarray(dists_nearest_neighbors) > limit_2)
		# find those under the limit
		# count them
		temp_count_1 = np.count_nonzero(np.asarray(dists_nearest_neighbors) < limit_1)
		temp_count_2 = np.count_nonzero(np.asarray(dists_nearest_neighbors) < limit_2)
		
		# return a list of counts across the frames
		outlier_counts_at_1m_across_frames.append(temp_count_1)
		outlier_counts_at_2m5_across_frames.append(temp_count_2)
		print("temp_count_1: {}".format(temp_count_1))
		print("temp_count_2: {}".format(temp_count_2))
	'''	
	
	outlier_counts_for_nn_across_frames = np.asarray(results_recording_dict["num_outliers_nn"])
	#total_landmarks_in_every_frame = results_recording_dict["num_clusters"][0]
	cluster_counts_for_nn_across_frames = np.asarray(results_recording_dict["num_clusters"])
	
	# find min/max/average/median of # of outliers
	#percent_outliers = (np.asarray(outlier_counts_for_nn_across_frames)/total_landmarks_in_every_frame)*100
	percent_outliers = (np.asarray(outlier_counts_for_nn_across_frames)/np.asarray(cluster_counts_for_nn_across_frames))*100
	
	min_nn_outliers = np.amin(percent_outliers)
	max_nn_outliers = np.amax(percent_outliers)
	avg_nn_outliers = np.average(percent_outliers)
	std_nn_outliers = np.std(percent_outliers)
	med_nn_outliers = np.median(percent_outliers)
	
	#tempstring = "min/max/avg/stddev/median percent outliers according to mutual-nn: {}/{}/{}/{}/{} ({} landmarks per frame)".format(min_nn_outliers,max_nn_outliers,avg_nn_outliers,std_nn_outliers,med_nn_outliers,total_landmarks_in_every_frame)
	tempstring = "min/max/avg/stddev/median percent outliers according to mutual-nn: {}/{}/{}/{}/{}".format(min_nn_outliers,max_nn_outliers,avg_nn_outliers,std_nn_outliers,med_nn_outliers)
	print(tempstring)
	measure_logfile.write("{}\n".format(tempstring))
	
	# generate histogram plot of outlier counts
	plt.figure()
	plt.hist(percent_outliers, bins=20)
	plt.title("Percentage of Outliers Based on Nearest-Neighbor Relationships")				# can add a note to the corner of the graph later what method it is, based on what method-run folder it's in
	plt.xlabel("Outlier Landmarks Percentage")
	plt.ylabel("Pointcloud Count")
	plt.xlim([0, 100])
	temp_plotpath = "{}/nn_precent_outliers_histogram.png".format(cluster_output_folder_path)
	plt.savefig(temp_plotpath)
	#if args.skip_graphs != True:
	#	plt.show()
	
	
	#------------------------------------------------------------------------------------------
	
	# collect the distances of mutual-nns across all frames into a list
	
	all_nn_distances = []
	for i in range(len(results_recording_dict["dists_nearest_neighbors"])):
		# get nn dists for this frame
		temp_nn_dists = results_recording_dict["dists_nearest_neighbors"][i]
		
		# concatenate onto aggregate list
		all_nn_distances.extend(temp_nn_dists)
		
	all_nn_distances = np.asarray(all_nn_distances)
	
	# find min/max/average/median of mutual-nn distances
	min_nn_dist = np.amin(all_nn_distances)
	max_nn_dist = np.amax(all_nn_distances)
	avg_nn_dist = np.average(all_nn_distances)
	std_nn_dist = np.std(all_nn_distances)
	med_nn_dist = np.median(all_nn_distances)
	std_nn_dist_real = find_halfnormal_stddev(all_nn_distances)
	
	tempstring = "min/max/avg/stdev/median/corrected_stddev distance between mutual-nn: {}/{}/{}/{}/{}/{}".format(min_nn_dist,max_nn_dist,avg_nn_dist,std_nn_dist,med_nn_dist,std_nn_dist_real)
	print(tempstring)
	measure_logfile.write("{}\n".format(tempstring))
	
	# generate histogram plot of mutual-nn distances
	plt.figure()
	plt.hist(all_nn_distances[(all_nn_distances<10)], bins=40)					#TODO added this limit to clip at 10m for graph consistency
	plt.title("Distance Between Mutual Nearest-Neighbor Landmarks")				# can add a note to the corner of the graph later what method it is, based on what method-run folder it's in
	plt.xlabel("Distance in Meters")
	plt.ylabel("Landmark Count")
	plt.xlim([0, 10])
	temp_plotpath = "{}/nn_distance_between_histogram.png".format(cluster_output_folder_path)
	plt.savefig(temp_plotpath)
	#if args.skip_graphs != True:
	#	plt.show()
	
	#------------------------------------------------------------------------------------------
	
	# collect the distances of all closest landmarks across all frames into a list
	
	all_closest_distances = []
	for i in range(len(results_recording_dict["dists_to_closest"])):
		# get closest landmark dists for this frame
		temp_closest_dists = results_recording_dict["dists_to_closest"][i]
		
		# concatenate onto aggregate list
		all_closest_distances.extend(temp_closest_dists)
	
	all_closest_distances = np.asarray(all_closest_distances)
		
	# find min/max/average/median of closest landmark distances
	min_closest_dist = np.amin(all_closest_distances)
	max_closest_dist = np.amax(all_closest_distances)
	avg_closest_dist = np.average(all_closest_distances)
	std_closest_dist = np.std(all_closest_distances)
	med_closest_dist = np.median(all_closest_distances)
	med_closest_dist_real = find_halfnormal_stddev(all_closest_distances)
	
	tempstring = "min/max/avg/stddev/median/corrected_stddev distance to closest landmark: {}/{}/{}/{}/{}/{}".format(min_closest_dist,max_closest_dist,avg_closest_dist,std_closest_dist,med_closest_dist,med_closest_dist_real)
	print(tempstring)
	measure_logfile.write("{}\n".format(tempstring))
	
	def plot_loghist(x, bins):
		hist, bins = np.histogram(x, bins=bins)
		logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
		plt.hist(x, bins=logbins)
		plt.xscale('log')
	
	# generate histogram plot of closest landmark distances
	plt.figure()
	plt.hist(all_closest_distances, bins=45)
	plt.title("Distance to the Nearest Landmark")				# can add a note to the corner of the graph later what method it is, based on what method-run folder it's in
	plt.xlabel("Distance in Meters")
	plt.ylabel("Landmark Count")
	temp_plotpath = "{}/nearest_landmark_dist_histogram.png".format(cluster_output_folder_path)
	plt.savefig(temp_plotpath)
	#if args.skip_graphs != True:
	#	plt.show()
	
	# zoomed in on 0-10 meters
	plt.figure()
	plt.hist(all_closest_distances[(all_closest_distances<10)], bins=40)
	plt.title("Distance to the Nearest Landmark (10 Meter Focus)")				# can add a note to the corner of the graph later what method it is, based on what method-run folder it's in
	plt.xlabel("Distance in Meters")
	plt.ylabel("Landmark Count")
	plt.xlim([0, 10])
	temp_plotpath = "{}/nearest_landmark_dist_histogram_10_focus.png".format(cluster_output_folder_path)
	plt.savefig(temp_plotpath)
	if args.skip_graphs != True:
		plt.show()
	
	# logarithmic version
	'''
	plt.figure()
	plot_loghist(all_closest_distances, 45)
	plt.title("Distance to the Nearest Landmark (Log)")				# can add a note to the corner of the graph later what method it is, based on what method-run folder it's in
	plt.xlabel("Distance in Meters")
	plt.ylabel("Landmark Count")
	if args.skip_graphs != True:
		plt.show()
	'''	
	
	measure_logfile.close()
	
	# dict to record lists of statistics over the sequence
	#results_recording_dict = {
	#	"frame_1_max":[],			# the maximum limits of points in the frame
	#	"frame_1_min":[],
	#	"frame_2_max":[],
	#	"frame_2_min":[],
	#	"num_points_1":[],			# the number of points in the frame. together with the bounds of the bounding box above, rough density can be computed
	#	"num_points_2":[],
	#	
	#	"num_clusters" : [],		# should be the same for both frames, technically for all frames
	#	"num_outliers_nn" : [],		# number of cluster points not in a mutual-nn relationship. should be same between sets, see below
	#	"dists_nearest_neighbors" : [],	# for those in a nn relationship, how far apart are they?
	#	
	#	"dists_to_closest" : []		# record all the distance to the nearest point for all points in frame 1, we can apply range gating later, or just plot a distribution
	#								# we only record the number of outliers that don't have a pairing within 1m in frame 1, we'll get to the reverse pairing where we consider frame 2 later in the sequence
	#}

elif (mode == "test_real_and_synthetic_landmark_association") or (mode == "test_real_and_synthetic_landmark_association_grassgraph_data"):
	print("time to print stats for this pass of landmark stats!")
	
	#alignment_output_folder_path -> where to save stats
	print("saving results to: {}".format(alignment_output_folder_path))
	if not os.path.exists(alignment_output_folder_path):
	    os.makedirs(alignment_output_folder_path)
	
	
	logfile_path = "{}/logfile.txt".format(alignment_output_folder_path)
	print("results logfile path: {}".format(logfile_path))
	
	measure_logfile = open(logfile_path, 'a')
	measure_logfile.write("\n")
	
	append_results_csv_path = "{}/append_collected_stats.csv".format(alignment_output_folder_path)
	tempstring = "current date and time: {}".format(datetime.datetime.now())
	print(tempstring)
	measure_logfile.write("{}\n".format(tempstring))
	tempstring = "csv output file path: {}".format(append_results_csv_path)
	print(tempstring)
	measure_logfile.write("{}\n".format(tempstring))
	
	append_results_csv_file = open(append_results_csv_path, 'a')
	
	'''
	results_recording_dict_2 = {
		"num_matches_list":					[],
		"inlier_frob_norm_list":			[],
		"fullset_frob_norm_list":			[],
		"num_matches_total_possible_list":	[],
		"matrix_frob_list":					[],
		"rotation_difference_angle_list":	[]
	}
	'''
	
	#------------------------------------------------------------------------------------------
	# what to store for each combination of settings
	#	percent outliers
	#	noise standard deviation
	#	number of inliers found statistics
	#	inlier frob norm statistics
	#	fullset frob norm statistics
	# generate histograms for those three if you like
	#
	
	# write a line with the results to the CSV
	
	num_matches_array = np.asarray(results_recording_dict_2["num_matches_list"])
	inlier_frob_norm_array = np.asarray(results_recording_dict_2["inlier_frob_norm_list"])
	fullset_frob_norm_array = np.asarray(results_recording_dict_2["fullset_frob_norm_list"])
	num_matches_total_possible_array = np.asarray(results_recording_dict_2["num_matches_total_possible_list"])
	matrix_frob_array = np.asarray(results_recording_dict_2["matrix_frob_list"])
	rotation_difference_angle_array = np.asarray(results_recording_dict_2["rotation_difference_angle_list"])
	
	# compute inliers found by grassgraph as a percentage of the number of landmark points
	percent_matches_array = np.divide(num_matches_array, num_matches_total_possible_array)
	
	# compute stats for this run and save to file
	percent_matches_stats_string = "{},{},{},{},{}".format(
		np.amin(percent_matches_array),
		np.amax(percent_matches_array),
		np.average(percent_matches_array),
		np.std(percent_matches_array),
		np.median(percent_matches_array)
	)
	
	inlier_frob_norm_stats_string = "{},{},{},{},{}".format(
		np.amin(inlier_frob_norm_array),
		np.amax(inlier_frob_norm_array),
		np.average(inlier_frob_norm_array),
		np.std(inlier_frob_norm_array),
		np.median(inlier_frob_norm_array)
	)
	
	fullset_frob_norm_stats_string = "{},{},{},{},{}".format(
		np.amin(fullset_frob_norm_array),
		np.amax(fullset_frob_norm_array),
		np.average(fullset_frob_norm_array),
		np.std(fullset_frob_norm_array),
		np.median(fullset_frob_norm_array)
	)
	
	matrix_frob_norm_stats_string = "{},{},{},{},{}".format(
		np.amin(matrix_frob_array),
		np.amax(matrix_frob_array),
		np.average(matrix_frob_array),
		np.std(matrix_frob_array),
		np.median(matrix_frob_array)
	)
	
	rotation_difference_angle_stats_string = "{},{},{},{},{}".format(
		np.amin(rotation_difference_angle_array),
		np.amax(rotation_difference_angle_array),
		np.average(rotation_difference_angle_array),
		np.std(rotation_difference_angle_array),
		np.median(rotation_difference_angle_array)
	)
	
	append_results_csv_file.write("{},{},{},{},{},{},{},{}\n".format(
		args.sequence,
		args.gen_pcent_ouliers,
		args.std_dev_noise,
		percent_matches_stats_string,
		inlier_frob_norm_stats_string,
		fullset_frob_norm_stats_string,
		matrix_frob_norm_stats_string,
		rotation_difference_angle_stats_string
	))
	
	# log settings and the column labels
	tempstring = "wrote a line to {} with {} sequence number {} percent outliers and {} std deviation noise, folowing headings:".format(args.sequence,append_results_csv_path,args.gen_pcent_ouliers,args.std_dev_noise)
	print(tempstring)
	measure_logfile.write("{}\n".format(tempstring))
	tempstring = "seq number, pcent outliers setting, std dev setting, pcent matches (min/max/avg/std/med), inlier frobs (min/max/avg/std/med), fullset frobs (min/max/avg/std/med), matrix frobs (min/max/avg/std/med), rotation angle differences (min/max/avg/std/med)"
	print(tempstring)
	measure_logfile.write("{}\n".format(tempstring))
	
	append_results_csv_file.close()
	measure_logfile.close()
	
	#assert 0, "unimplemented"
	
	print("bad_matrix_count: {}".format(bad_matrix_count))
	print("matrix_count: {}".format(matrix_count))


'''
# visualize every 100th scan
for i in range(len(out_dict["frame_id"])):
	frame_id = out_dict["frame_id"][i]
	scan_points = out_dict["scan_points"][i] #TODO should be scan_points
	frame_xyz = out_dict["frame_pose"][i]
	
	# visualize every 100th scan
	if (not (i % 100)):
		print("number of points in frame {}: {}".format(frame_id,scan_points.shape))
		pcd = o3d.geometry.PointCloud()
		pcd.points = o3d.utility.Vector3dVector(scan_points)
		#coord = o3d.geometry.TriangleMesh().create_coordinate_frame(size=20.0)
		coord1 = o3d.geometry.TriangleMesh().create_coordinate_frame(size=10.0,origin=frame_xyz)
		o3d.visualization.draw_geometries([pcd,coord1])
'''


























