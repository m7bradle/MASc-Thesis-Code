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
from sklearn.cluster import DBSCAN
from sklearn.cluster import Birch
import keyboard

import matlab.engine

#---------------------------------------------------------------------------

ROAD_LABEL = 0
SIDEWALK_LABEL = 1
BUILDING_LABEL = 2
VEGETATION_LABEL = 8
CAR_LABEL = 13

#turned this off, turn it on again before trying the grassmannian !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# TODO this is now on by default, except for when it gets turned off for measurements of the clusters themselves that need total correction turned off
# should we add/subtract points to get the target_num number of clusters?
#correct_num_clusters = True
#correct_num_clusters = False

#method_debug = False
method_debug = True


#input: scan_points, target_num, rng, scan_labels
#output: point_labels, cluster_points
def do_clustering(in_dict, method="cluster", remove_noise=False, eng=None, correct_num_clusters=True):
	if method == "cluster":
		#my_points = in_dict["scan_points"]
		#my_labels = in_dict["scan_labels"]
		#target_num = in_dict["target_num"]
		#rng = in_dict["rng"]
		
		my_points = in_dict["scan_points"]
		#print("my_points.shape: {}".format(my_points.shape))
		
		if "scan_labels" in in_dict:
			my_labels = in_dict["scan_labels"]
		else:
			my_labels = None
		
		if "which_clustering_override" in in_dict:
			which_clustering_override = in_dict["which_clustering_override"]
		else:
			which_clustering_override = None
		
		if "target_num" in in_dict:
			target_num = in_dict["target_num"]
		else:
			target_num = None
		
		if "rng" in in_dict:
			rng = in_dict["rng"]
		else:
			rng = None
		
		if remove_noise:
			pcd = o3d.geometry.PointCloud()
			pcd.points = o3d.utility.Vector3dVector(my_points)
			cloud, ind = pcd.remove_radius_outlier(nb_points=2, radius=2.5)
			if my_labels is not None:
				my_points = np.take(my_points, ind, axis=0)
				my_labels = np.take(my_labels, ind, axis=0)
			else:
				my_points = np.take(my_points, ind, axis=0)
		
		'''
		print("points: {} labels: {}".format(my_points.shape,(my_labels.shape if my_labels is not None else "None")))
		# an awful hack, but I'm sick of this (eg in birch clustering or amin/amax) *sometimes* dying because of this, unpredictably
		# TODO untested? the heisenbug I was experienceing went away after en environment reinstall
		if my_points.shape[0] == 0:
			print("there are no points for clustering! we're making some up! they're almost certainly in the wrong place too, in addition to being random!")
			lower_xyz_bound = np.asarray([-5.0, 0.0, -5.0]) # this is totally imaginary, has little basis in reality, and almost certainly isn't centered on the current frame
			upper_xyz_bound = np.asarray([5.0, 5.0, -5.0])	# however, it's the best we've got and it's not like a frame with zero points is expected to match against anything anyway
			my_points = rng.uniform(low=lower_xyz_bound, high=upper_xyz_bound, size=(120,3))
			if my_labels is not None:
				my_labels = np.full((my_points.shape[0],1), -1)
			#assert(0)
		'''
		#-----------------------------------------------------------------------
		# my_points, my_labels
		
		# if semantic labels are provided
		if my_labels is not None:
			
			# major classes: road,sidewalk,building,vegetation,car
			# suggest: building,vegetation,car
			
			if method_debug:
				print("clustering debug: doing clustering with semantic data")
			
			
			if which_clustering_override is not None:
				if which_clustering_override == "birch":
					single_pass_layered_clustering = True #BIRCH approach
				elif which_clustering_override == "dbscan":
					single_pass_layered_clustering = False #DBSCAN approach
				else:
					assert 0, "not a valid clustering method to be selected via override!"
			else:
				if method_debug:
					print("clustering debug: running 'default' hardcoded semantic handling technique")
				# stack semantic classes into offset layers, or do clustering on semantic classes individually?
				single_pass_layered_clustering = True #BIRCH approach
				#single_pass_layered_clustering = False #DBSCAN approach

			# should we exclude any classes?
			exclude_building = False
			#exclude_building = True
			exclude_vegetation = False
			#exclude_vegetation = True
			exclude_car = False
			#exclude_car = True
			
			# for each class above, get the points that belong to each class
			mask_building = (my_labels == BUILDING_LABEL)
			mask_vegetation = (my_labels == VEGETATION_LABEL)
			mask_car = (my_labels == CAR_LABEL)
			
			points_building = my_points[mask_building]
			points_vegetation = my_points[mask_vegetation]
			points_car = my_points[mask_car]
			
			#print("my_points.shape: {} my_labels.shape: {}".format(my_points.shape, my_labels.shape))
			#print("points_building.shape: {} points_vegetation.shape: {} points_car.shape: {}".format(points_building.shape, points_vegetation.shape, points_car.shape))
			
			# if doing stacking method
			if single_pass_layered_clustering:
				if method_debug:
					print("clustering debug: layered clustering with all semantic classes at once (uses birch)")
				
				# for DSO with an upright camera, upwards is negative Y because positive Z is forwards
				# using 45 meters for good separation because scan radius is 45 meters
				offset_shift = np.asarray([0.0,-45.0,0.0])*1 #TODO maybe double or tripple this, since 45 is the radius, not the diameter. also double check Z direction?
				
				# apply specific shift along negative Y to each set of points
				offset_points_building = points_building + (offset_shift*1.0)
				offset_points_vegetation = points_vegetation + (offset_shift*2.0)
				offset_points_car = points_car + (offset_shift*3.0)
				#print("{} => {} {} {}".format(my_points.shape,offset_points_building.shape,offset_points_vegetation.shape,offset_points_car.shape))
				
				# concatenate together
				concat_points = np.concatenate((offset_points_building,offset_points_vegetation,offset_points_car), axis=0)
				
				if target_num is None:
					temp_target_num = 120
				else:
					temp_target_num = target_num
				
				# do (BIRCH) clustering
				#clustering = DBSCAN(eps=0.75, min_samples=5).fit(concat_points)
				clustering = Birch(threshold=0.5, n_clusters=temp_target_num).fit(concat_points)
				cluster_gen_labels = clustering.labels_
				cluster_gen_points = np.concatenate((points_building,points_vegetation,points_car), axis=0)
				
				# making sure that the number of points used to generate cluster labels and the number of coord points (in selected semantic classes) are the same
				# (two paths concatenating points with selected labels, with and without shifting)
				assert concat_points.shape == cluster_gen_points.shape
				
				#print("number of clusters found through iterative layered BIRCH is: {}".format(np.amax(cluster_gen_labels)+1))
				
			# if doing repeated clustering for each semantic class
			else:
				if method_debug:
					print("clustering debug: repeated clustering for each semantic class (uses dbscan)")
				
				#clustering = DBSCAN(eps=0.75, min_samples=5).fit(concat_points)
				#clustering = Birch(threshold=0.5, n_clusters=120).fit(concat_points)
				
				#eps_value = 0.75
				#min_samples_value = 5
				#eps_value = 3.0
				#min_samples_value = 25
				
				#eps_value_building = 2.0
				eps_value_building = 3.0
				min_samples_value_building = 25
				eps_value_vegetation = 3.0
				min_samples_value_vegetation = 35
				#eps_value_car = 3.0
				#eps_value_car = 0.75
				#eps_value_car = 1.5
				eps_value_car = 2.0
				min_samples_value_car = 25
				
				'''
				clustering debug: doing clustering with semantic data
				clustering debug: repeated clustering for each semantic class (uses dbscan)
				Traceback (most recent call last):
				  File "./pointcloud_cluster_testing_measurement.py", line 1358, in <module>
					cluster_dict = do_clustering({"scan_points": scan_points2,"scan_labels":scan_labels2,"target_num":TARGET_NUM, "which_clustering_override":which_clustering_override}, method=method_string, remove_noise=True, eng=eng)
				  File "/home/matt/Documents/waterloo/spatial_vpr/grass_association/grass_assoc_testing/points_clustering_helper.py", line 190, in do_clustering
					clustering = DBSCAN(eps=eps_value_car, min_samples=min_samples_value_car).fit(points_car)
				  File "/home/matt/anaconda3/envs/grass_assoc_testing2/lib/python3.8/site-packages/sklearn/cluster/_dbscan.py", line 368, in fit
					X = self._validate_data(X, accept_sparse="csr")
				  File "/home/matt/anaconda3/envs/grass_assoc_testing2/lib/python3.8/site-packages/sklearn/base.py", line 546, in _validate_data
					X = check_array(X, input_name="X", **check_params)
				  File "/home/matt/anaconda3/envs/grass_assoc_testing2/lib/python3.8/site-packages/sklearn/utils/validation.py", line 931, in check_array
					raise ValueError(
				ValueError: Found array with 0 sample(s) (shape=(0, 3)) while a minimum of 1 is required by DBSCAN.
				'''
				
				print("my points shape: {} my labels shape: {}".format(my_points.shape,my_labels.shape))
				print("number of building points: {} vegetation points: {} car points: {}".format(points_building.shape,points_vegetation.shape,points_car.shape))
				
				# do (DBSCAN) clustering on each set of points
				# labels points as in cluster 0, 1, 2... except points that are in no clusters, which are labeled -1
				# if there are no points for one of these classes, eg there are no points from cars, then replace with a zero-length list of labels
				if points_building.shape[0] > 0:
					clustering = DBSCAN(eps=eps_value_building, min_samples=min_samples_value_building).fit(points_building)
					cluster_labels_building = clustering.labels_
				else:
					cluster_labels_building = np.full((0), fill_value=-1) # list of point labels of length zero
				if points_vegetation.shape[0] > 0:
					clustering = DBSCAN(eps=eps_value_vegetation, min_samples=min_samples_value_vegetation).fit(points_vegetation)
					cluster_labels_vegetation = clustering.labels_
				else:
					cluster_labels_vegetation = np.full((0), fill_value=-1) # list of point labels of length zero
				if points_car.shape[0] > 0:
					clustering = DBSCAN(eps=eps_value_car, min_samples=min_samples_value_car).fit(points_car)
					cluster_labels_car = clustering.labels_
				else:
					cluster_labels_car = np.full((0), fill_value=-1) # list of point labels of length zero
				
				# not needed, output of clustering is already numpy array
				#cluster_labels_building = np.asarray(cluster_labels_building)
				#cluster_labels_vegetation = np.asarray(cluster_labels_vegetation)
				#cluster_labels_car = np.asarray(cluster_labels_car)

				
				#print("labels shapes: {} {} {}".format(cluster_labels_building.shape, cluster_labels_vegetation.shape, cluster_labels_car.shape))
				
				# offset the cluster labels of each set
				combined_labels = cluster_labels_building.copy()
				if exclude_building:
					combined_labels = np.full(cluster_labels_building.shape, fill_value=-1)

				# if the combined list of labels is still of length zero, then we don't need to lift any clusters (eg. cluster zero) above anything, so the amount to add is zero
				if combined_labels.shape[0] > 0:
					add_amount = np.amax(combined_labels)+1
				else:
					add_amount = 0
				temp_labels = cluster_labels_vegetation.copy()
				temp_labels[temp_labels != -1] += add_amount			# offset so that zero labels are above the maximum value of the previous set
				if exclude_vegetation:
					temp_labels = np.full(cluster_labels_vegetation.shape, fill_value=-1)
				combined_labels = np.concatenate((combined_labels,temp_labels), axis=0)	# now that all cluster labels' values are shifted above existing cluster labels, it's safe to concatenate

				# if the combined list of labels is still of length zero, then we don't need to lift any clusters (eg. cluster zero) above anything, so the amount to add is zero
				if combined_labels.shape[0] > 0:
					add_amount = np.amax(combined_labels)+1
				else:
					add_amount = 0
				temp_labels = cluster_labels_car.copy()
				temp_labels[temp_labels != -1] += add_amount			# offset so that zero labels are above the maximum value of the previous set
				if exclude_car:
					temp_labels = np.full(cluster_labels_car.shape, fill_value=-1)
				combined_labels = np.concatenate((combined_labels,temp_labels), axis=0)	# now that all cluster labels' values are shifted above existing cluster labels, it's safe to concatenate


				'''
				combined_labels = cluster_labels_building.copy()
				if exclude_building:
					combined_labels = np.full(cluster_labels_building.shape, fill_value=-1)
				
				temp_labels = cluster_labels_vegetation.copy()
				temp_labels[temp_labels != -1] += (np.amax(combined_labels)+1)			# offset so that zero labels are above the maximum value of the previous set
				if exclude_vegetation:
					temp_labels = np.full(cluster_labels_vegetation.shape, fill_value=-1)
				combined_labels = np.concatenate((combined_labels,temp_labels), axis=0)	# now that all cluster labels' values are shifted above existing cluster labels, it's safe to concatenate
				
				temp_labels = cluster_labels_car.copy()
				temp_labels[temp_labels != -1] += (np.amax(combined_labels)+1)			# offset so that zero labels are above the maximum value of the previous set
				if exclude_car:
					temp_labels = np.full(cluster_labels_car.shape, fill_value=-1)
				combined_labels = np.concatenate((combined_labels,temp_labels), axis=0)	# now that all cluster labels' values are shifted above existing cluster labels, it's safe to concatenate
				'''
				#print("labels shapes: {} {} {} => {}".format(cluster_labels_building.shape, cluster_labels_vegetation.shape, cluster_labels_car.shape, combined_labels.shape))
				
				# concatenate points and cluster labels
				cluster_gen_labels = combined_labels
				cluster_gen_points = np.concatenate((points_building,points_vegetation,points_car), axis=0)
				
				#print("coords shapes: {} {} {} => {}".format(points_building.shape, points_vegetation.shape, points_car.shape, cluster_gen_points.shape))
				
				# ensuring the same number of cluster labels (put together in stages) as coord points with the selected semantic labels (should be 1:1 mapping)
				assert cluster_gen_labels.shape[0] == cluster_gen_points.shape[0]
				
				#TODO
				if cluster_gen_points.shape[0] > 0:
					print("number of clusters found through iterative DBSCAN is: {}".format(np.amax(cluster_gen_labels)+1))
				else:
					print("number of clusters found through iterative DBSCAN is: zero, because there are no cluster points found by dbscan")
				
			# points with undesirable semantic labels are excluded, cluster labels generated
			
			
		else:
			if method_debug:
				print("clustering debug: doing clustering with *no* semantic data")
			
			# semantic labels are not provided
			# do regular dbscan/birch clustering
			# list of points unchanged, cluster labels generated
			
			if which_clustering_override is not None:
				if which_clustering_override == "birch":
					if method_debug:
						print("clustering debug: doing regular birch")
					
					clustering = Birch(threshold=0.5, n_clusters=120).fit(my_points)
					
				elif which_clustering_override == "dbscan":
					if method_debug:
						print("clustering debug: doing regular dbscan")
					
					clustering = DBSCAN(eps=0.75, min_samples=5).fit(my_points)
					
				else:
					assert 0, "not a valid clustering method to be selected via override!"
			else:
				if method_debug:
					print("clustering debug: running 'default' hardcoded clustering technique")
				
				#clustering = DBSCAN(eps=0.75, min_samples=5).fit(my_points)
				clustering = Birch(threshold=0.5, n_clusters=120).fit(my_points)
			
			cluster_gen_labels = clustering.labels_
			cluster_gen_points = my_points
		
		#-----------------------------------------------------------------------
		# some (cluster) points, some cluster labels
		
		#correct_num_clusters = True
		cluster_points = generate_cluster_points(cluster_gen_points, cluster_gen_labels, target_num, rng, correct_num_clusters)
		
		out_dict = {}
		out_dict["scan_points"] = cluster_gen_points
		out_dict["point_labels"] = cluster_gen_labels
		out_dict["cluster_points"] = cluster_points
		return out_dict
		
		
	elif method == "pcsimp":
		my_points = in_dict["scan_points"]
		#print("my_points.shape: {}".format(my_points.shape))
		
		alpha_param = 0.1;		# 0.1 -> 10%, for 150/9469(?) need 0.026401943 (2.64%); average is 9176 -> 0.027244987
		#alpha_param = 0.027244987
		#lambda_param = 1e-3;
		#lambda_param = 1e-5;
		lambda_param = 0.0;
		#K_param = 15;
		#p_thres_min_param = 3000;
		#p_thres_max_param = 8000;
		
		out_dict = eng.pcsimp_wrapper({"X":matlab.double(my_points),"alpha":alpha_param,"lambda":lambda_param})
		simplified_points = np.asarray(out_dict["simpX"])
		print("simplified_points.shape: {}".format(simplified_points.shape))
		
		out_dict = {}
		#out_dict["scan_points"] = my_points
		out_dict["cluster_points"] = simplified_points
		return out_dict
		
	else:
		assert(0)




# in future may take frame_id or dso frame index if semantic segmentation is employed

#TODO orphaned function? kept for posterity or incase soembody is using it somewhere? pretty sure it's only used here
# takes:	scan_points (the points in this keyframe's scan)
# returns:	numeric label for each point in scan_points. -1 means unclustered (outlier or rejected)
def label_points(scan_points):
	
	#DBSCAN(eps=1, min_samples=5)
	#DBSCAN(eps=0.5, min_samples=5)
	my_points = scan_points
	
	clustering = DBSCAN(eps=0.75, min_samples=5).fit(my_points)
	#clustering = Birch(threshold=0.5, n_clusters=120).fit(my_points)
	labels = clustering.labels_
	
	'''
	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(scan_points)
	points_out = np.asarray(pcd.voxel_down_sample(voxel_size=3).points)
	#points_out = np.asarray(pcd.voxel_down_sample(voxel_size=10).points)
	labels = np.ones((points_out.shape[0]),dtype="int")
	print("number of points in downsampled point cloud: {}".format(points_out.shape[0]))
	'''
	
	return labels


# the below function only produces clusters where 1 cluster label = 1 cluster point
# if you have semantically labled data you need to take each set of points with a particular semantic label and run clustering on it
# then those labled instance clusters can be converted to single points with the function below, and combined to make the full set for grassmannian matching

#TODO pretty sure it's only used in this file, by do_clustering() above

# in future may take frame_id or dso frame index if semantic segmentation is employed
# takes:	scan_points (the points in this keyframe's scan), point_labels (their cluster labels), target_num (the required number of points for the grassmannian representation)
# returns:	the clusters from the above function, turned into single points for grassmannian matching
def generate_cluster_points(scan_points, point_labels, target_num=None, rng=None, correct_num_clusters=True):
	if rng is None:
		rng = np.random.default_rng(seed=0)
	if target_num is None:
		target_num = 120
	
	#point_labels = label_points(scan_points)
	label_values = np.unique(point_labels)
	#print("number of labels: {}".format(label_values.shape[0]))
	
	# collect points for each label and compute their average position to produce a single representative point
	cluster_points = []
	for val in label_values:
		if val != -1:
			indexes = np.argwhere(point_labels==val)[:,0] #input is 1D, so take the first dimension
			single_clusters_points = np.take(scan_points, indexes, axis=0)
			cluster_center = np.average(single_clusters_points, axis=0)
			cluster_points.append(cluster_center)
	cluster_points = np.asarray(cluster_points)
	
	# ignore unclustered points when counting clusters
	unclustered_indexes = np.argwhere(label_values==-1)
	num_raw_clusters = np.delete(label_values, unclustered_indexes).shape[0]
	#print("number of raw clusters: {}".format(num_raw_clusters))
	
	# the grassmannian association implementation we're using requires that
	# every collection of points we association has the same number of points
	
	if correct_num_clusters:
		
		# if there is a significant difference in the number of clusters from what's desired, print a warning
		if np.abs(num_raw_clusters - target_num) > 20:
			if num_raw_clusters < 5:
				print("number of scan points: {}".format(scan_points.shape))
				print("Warning! number of raw clusters {} is off from required number of {} and there are less than 5 in total ------------------------------------------------".format(num_raw_clusters,target_num))
			else:
				print("number of scan points: {}".format(scan_points.shape))
				if num_raw_clusters < target_num:
					print("Warning! number of raw clusters {} is off from required number of {} by more than 20 (-) ----------------------------".format(num_raw_clusters,target_num))
				else:
					print("Warning! number of raw clusters {} is off from required number of {} by more than 20 (+)".format(num_raw_clusters,target_num))
		
		# add or remove some clusters to reach the desired number
		if num_raw_clusters > target_num:
			# pick some random cluster points to remove
			num_to_remove = num_raw_clusters - target_num
			print("removing {} clusters to reach {}".format(num_to_remove,target_num))
			remove_indexes = rng.choice(num_raw_clusters, size=(num_to_remove), replace=False)
			cluster_points = np.delete(cluster_points, remove_indexes, axis=0)
		elif num_raw_clusters < target_num:
			# generate some new random cluster points within the same range of values
			num_to_add = target_num - num_raw_clusters
			print("adding {} clusters to reach {}".format(num_to_add,target_num))
			if num_raw_clusters < 5:
				print("adding {} clusters to reach {}".format(num_to_add,target_num))
				# handle the case where few (or zero) clusters are found fo some weird scans. hopefully we have some scan points
				# in this case we just replace everything with target_num random clusters to handle the case where sometimes no clusters at all are found
				# it's maybe not the best way to handle this case, but scans with so few clusters are rare and probably won't be matched sucessfully anyway
				print("low clusters: {} num_scanpoints: {}, doing replacement with random clusters".format(num_raw_clusters, scan_points.shape))
				if scan_points.shape[0] == 0:
					# if there are zero points in the scan, there's really not much we can do
					# we don't even know where to make fake points to try to keep things moving
					# best we can do is like, make them at (0,0,0)?
					# ultimately, allowing empty scans, or even scans with too few points, was a poor implementation decision and something to learn from but kinda impossible to fix at this point
					print("no points! making up bounds")
					lower_xyz_bound = np.asarray([-5.0, 0.0, -5.0]) # this is totally imaginary, has little basis in reality, and almost certainly isn't centered on the current frame
					upper_xyz_bound = np.asarray([5.0, 5.0, -5.0])	# however, it's the best we've got and it's not like a frame with zero points is expected to match against anything anyway
				else:
					lower_xyz_bound = np.amin(scan_points, axis=0)
					upper_xyz_bound = np.amax(scan_points, axis=0)
					#print("bounds: {} {}".format(lower_xyz_bound,upper_xyz_bound))
				cluster_points = rng.uniform(low=lower_xyz_bound, high=upper_xyz_bound, size=(target_num,3))
			else:
				lower_xyz_bound = np.amin(cluster_points, axis=0)
				upper_xyz_bound = np.amax(cluster_points, axis=0)
				samples = rng.uniform(low=lower_xyz_bound, high=upper_xyz_bound, size=(num_to_add,3))
				cluster_points = np.concatenate((cluster_points,samples), axis=0)
	else:
		# we need to ensure there are least SOME clusters (not zero), or other code breaks horribly
		if num_raw_clusters < 5:
			#print("adding {} clusters to reach {}".format(num_to_add,target_num))
			
			target_num = 5
			
			# handle the case where few (or zero) clusters are found fo some weird scans. hopefully we have some scan points
			# in this case we just replace everything with target_num random clusters to handle the case where sometimes no clusters at all are found
			# it's maybe not the best way to handle this case, but scans with so few clusters are rare and probably won't be matched sucessfully anyway
			print("low clusters: {} num_scanpoints: {}, we're not supposed to be adding/removing anything right now (turned off) but if there are few/none things break (especially at 0). replacing with random 5.".format(num_raw_clusters, scan_points.shape))
			if scan_points.shape[0] == 0:
				# if there are zero points in the scan, there's really not much we can do
				# we don't even know where to make fake points to try to keep things moving
				# best we can do is like, make them at (0,0,0)?
				# ultimately, allowing empty scans, or even scans with too few points, was a poor implementation decision and something to learn from but kinda impossible to fix at this point
				print("no points! making up bounds")
				lower_xyz_bound = np.asarray([-5.0, 0.0, -5.0]) # this is totally imaginary, has little basis in reality, and almost certainly isn't centered on the current frame
				upper_xyz_bound = np.asarray([5.0, 5.0, -5.0])	# however, it's the best we've got and it's not like a frame with zero points is expected to match against anything anyway
			else:
				lower_xyz_bound = np.amin(scan_points, axis=0)
				upper_xyz_bound = np.amax(scan_points, axis=0)
				#print("bounds: {} {}".format(lower_xyz_bound,upper_xyz_bound))
			cluster_points = rng.uniform(low=lower_xyz_bound, high=upper_xyz_bound, size=(target_num,3))
		
		
		
		
	#print()
	#print("cluster_points.shape: {}".format(cluster_points.shape))
	return cluster_points






















