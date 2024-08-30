import os
import stat
#import sys
import numpy as np
#import pandas as pd

#test_real_and_synthetic_landmark_association
#test_real_and_synthetic_landmark_association_grassgraph_data
'''
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 00 --skip_every_n 150 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association_grassgraph_data \
	--cluster_output ./collected_cluster_stats/test_method2/ \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.0 --std_dev_noise 0.0
'''

num_threads = 8
#num_threads = 1

#runfile_path_root = "./measure_assoc_perf_kitti"
runfile_path_root = "./measure_detailed_outliers_kitti"
#runfile_path_root = "./measure_assoc_perf_shrec"
#output_folder_name = "basic_analysis_kitti"
output_folder_name = "detailed_outliers_kitti"
#output_folder_name = "basic_analysis_shrec"
use_shrec_dataset = False
#use_shrec_dataset = True
#percent_outliers_range = [0.0,0.05,0.1,0.15,0.2,0.3,0.4,0.5] # normal percent outlier range for kitti and shrec
percent_outliers_range = [0.0,0.0084,0.017,0.025,0.034,0.042,0.05,0.059,0.067,0.075,0.084] # detailed look at low outliers for kitti
std_dev_noise_range = [0.0,0.15,0.3,0.45,0.6,0.75,0.9,1.05,1.2,1.35,1.5,1.875,2.25,2.625,3.0,3.375,3.75,4.125,4.5,4.875,5.25,5.625,6.0] # noise steps for kitti
#std_dev_noise_range = [0.0,0.059,0.118,0.177,0.236,0.295,0.354,0.414,0.473,0.532,0.591,0.739,0.886,1.034,1.182,1.329,1.477,1.625,1.772,1.92,2.068,2.216,2.363] # noise steps for shrec
seq_numbers = ["00","02","05","06"]
#seq_numbers = ["00"]
skip_every = 1



'''
#runfile_path_root = "./measure_assoc_perf"
#output_folder_name = "sixth_run_replicating_gentle_outlier_run_with_1p5_stddev_attempt_2"
#output_folder_name = "basic_analysis_with_grassgraph_points_120_per_shape_detailed_outliers"

#runfile_path_root = "./measure_assoc_perf_nonoise_regular"
#runfile_path_root = "./measure_assoc_perf_nonoise_zoomed"
#runfile_path_root = "./measure_assoc_perf_nonoise_regular_grassgraph"
#runfile_path_root = "./measure_assoc_perf_nonoise_zoomed_grassgraph"
#output_folder_name = "basic_analysis_with_no_noise_case"
#output_folder_name = "closeup_outliers_with_no_noise_case"
#output_folder_name = "basic_analysis_with_no_noise_case_grassgraph_data"
#output_folder_name = "closeup_outliers_with_no_noise_case_grassgraph_data"

#runfile_path_root = "./measure_assoc_perf_nonoise_regular_grassgraph_no120"
#output_folder_name = "basic_analysis_with_no_noise_case_grassgraph_data_not_only_120_points"

runfile_path_root = "./measure_assoc_perf_grassgraph"
output_folder_name = "basic_analysis_grassgraph"
#runfile_path_root = "./measure_assoc_perf_grassgraph_120"
#output_folder_name = "basic_analysis_grassgraph_120"
#runfile_path_root = "./measure_assoc_perf_grassgraph_120_zoomed"
#output_folder_name = "basic_analysis_grassgraph_120_zoomed"

skip_every = 1
#percent_outliers_range = [0.0,0.05,0.1,0.15,0.2,0.3,0.4,0.5]
percent_outliers_range = [0.0,0.0084,0.017,0.025,0.034,0.042,0.05,0.059,0.067,0.075,0.084] # detailed look at low end  #TODO

# wonky rainge basis #std_dev_noise_range = [0.238,0.476,0.714,0.952,1.19,1.428,1.666,1.904,2.142,2.38,2.975,3.57,4.165,4.76,5.355,5.95,6.545,7.14,7.735,8.33] #based on 2.38
# wonky rainge basis #std_dev_noise_range = [0.3,0.6,0.9,1.2,1.5,1.8,2.1,2.4,2.7,3.0,3.75,4.5,5.25,6.0,6.75,7.5,8.25,9.0,10.5] #based on 3.0
# missing zero #std_dev_noise_range = [0.15,0.3,0.45,0.6,0.75,0.9,1.05,1.2,1.35,1.5,1.875,2.25,2.625,3.0,3.375,3.75,4.125,4.5,4.875,5.25,5.625,6.0] #based on 1.5
# missing zero #std_dev_noise_range = [0.059,0.118,0.177,0.236,0.295,0.354,0.414,0.473,0.532,0.591,0.739,0.886,1.034,1.182,1.329,1.477,1.625,1.772,1.92,2.068,2.216,2.363] # based on shreck13, but applicable for both
#std_dev_noise_range = [0.0,0.15,0.3,0.45,0.6,0.75,0.9,1.05,1.2,1.35,1.5,1.875,2.25,2.625,3.0,3.375,3.75,4.125,4.5,4.875,5.25,5.625,6.0] #based on 1.5
std_dev_noise_range = [0.0,0.059,0.118,0.177,0.236,0.295,0.354,0.414,0.473,0.532,0.591,0.739,0.886,1.034,1.182,1.329,1.477,1.625,1.772,1.92,2.068,2.216,2.363] # based on shreck13, but applicable for both
#std_dev_noise_range = [0.0]

#std_dev_noise_range = [0.0,0.01,0.05,0.1,0.15,0.2] # -> should range from zero, to the measurement(s) made, to perhaps double those measurements
# paper: 0.0,0.01,0.05,0.1,0.15,0.2

#TODO testing
#percent_outliers_range = [0.0,0.1,0.2]
#std_dev_noise_range = [0.0,0.1,0.2,0.3,0.4,0.5]


seq_numbers = ["00","02","05","06"]
'''

count = 0
for i in range(len(percent_outliers_range)):
	for j in range(len(std_dev_noise_range)):
		count += 1
print("num combinations: {}".format(count))

command_prototype_string_kitti = \
"""time python ./pointcloud_cluster_testing_measurement.py \\
	--sequence {} --skip_every_n {} \\
	--skip_vis true --skip_graphs true \\
	--mode test_real_and_synthetic_landmark_association \\
	--use_semantic no --clustering_override birch --cluster_target_num 120 \\
	--gen_pcent_ouliers {} --std_dev_noise {} \\"""
#	--alignment_output ./collected_alignment_stats/{}/ # has to be added after

command_prototype_string_shrec = \
"""time python ./pointcloud_cluster_testing_measurement.py \\
	--sequence {} --skip_every_n {} \\
	--skip_vis true --skip_graphs true \\
	--mode test_real_and_synthetic_landmark_association_grassgraph_data \\
	--use_semantic no --clustering_override birch --cluster_target_num 120 \\
	--gen_pcent_ouliers {} --std_dev_noise {} \\"""
#	--alignment_output ./collected_alignment_stats/{}/ # has to be added after

if not use_shrec_dataset:
	command_prototype_string = command_prototype_string_kitti
else:
	command_prototype_string = command_prototype_string_shrec

list_of_titles = []
list_of_commands = []
for seq_num in seq_numbers:
	for i in range(len(percent_outliers_range)):
		for j in range(len(std_dev_noise_range)):
			# print("pcent: {} dev: {}".format(percent_outliers_range[i],std_dev_noise_range[j]))
			
			pcent_outliers_string = str(percent_outliers_range[i])
			std_dev_string = str(std_dev_noise_range[j])
			
			#command_string = command_prototype_string.format(seq_num,skip_every,output_folder_name,pcent_outliers_string,std_dev_string)
			command_string = command_prototype_string.format(seq_num,skip_every,pcent_outliers_string,std_dev_string)
			list_of_titles.append("{}\n".format("echo \"runfile: doing run for {} outliers and {} stddev on seq {}\"".format(pcent_outliers_string,std_dev_string,seq_num)))
			list_of_commands.append("{}\n".format(command_string))
			print("generated text for seq={} outlier={} stddev={}".format(seq_num,pcent_outliers_string,std_dev_string))
			

# shuffle before splitting so you don't end up with all the expensive Sequence 00 etc ones in the same group
rng = np.random.default_rng(5)

conjoined = np.asarray([list_of_titles,list_of_commands]).transpose()
rng.shuffle(conjoined, axis=0)
list_of_titles = conjoined[:,0].tolist()
list_of_commands = conjoined[:,1].tolist()

'''
# https://stackoverflow.com/questions/2130016/splitting-a-list-into-n-parts-of-approximately-equal-length
# presumably also broken
def list_split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

#TODO this code is broken!!!!!!! eg for chunkIt(range(8), 6) -> 7 lists in output!!!
def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0
    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg
    return out
'''
# solution based on suggested numpy function
split_list_of_titles = [x.tolist() for x in np.array_split(np.asarray(list_of_titles),num_threads)]
split_list_of_commands = [x.tolist() for x in np.array_split(np.asarray(list_of_commands),num_threads)]

#split_list_of_titles = chunkIt(list_of_titles, num_threads)
#split_list_of_commands = chunkIt(list_of_commands, num_threads)
assert (len(split_list_of_titles) == num_threads), "chunking made a list that doesn't match the threads!"
assert (len(split_list_of_commands) == num_threads), "chunking made a list that doesn't match the threads!"

print("num_threads: {}".format(num_threads))
for i in range(len(split_list_of_titles)):
	print("{}: title list length = {}".format(i, len(split_list_of_titles[i])))
for i in range(len(split_list_of_commands)):
	print("{}: command list length = {}".format(i, len(split_list_of_commands[i])))
#assert 0, "this assert is meant to trigger"




for thread_index in range(num_threads):
	
	runfile_path = "{}_{}.sh".format(runfile_path_root,thread_index)
	list_of_titles = split_list_of_titles[thread_index]
	list_of_commands = split_list_of_commands[thread_index]
	output_folder_path_part = "{}_{}".format(output_folder_name,thread_index)
	
	print("writing script to: {}".format(runfile_path))
	runfile_file = open(runfile_path, "w")
	runfile_file.write("{}\n".format("#!/bin/bash"))
	runfile_file.write("{}\n".format("set -e"))
	
	for i in range(len(list_of_titles)):
		runfile_file.write(list_of_titles[i])
		runfile_file.write(list_of_commands[i])
		runfile_file.write("\t--alignment_output ./results_output/collected_alignment_stats/{}/ \n".format(output_folder_path_part))
		runfile_file.write("echo \"runfile: done run {} of {} \"\n".format(i+1,len(list_of_titles)))

	runfile_file.write("echo \"end of runfile\"")
	runfile_file.close()

	# add execute permissions for everyone, equivalent to "chmod +x <file>"
	file_stat = os.stat(runfile_path)
	os.chmod(runfile_path, file_stat.st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
	print("done writing to {}".format(runfile_path))

















