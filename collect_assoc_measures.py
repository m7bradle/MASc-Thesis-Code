import os
import stat
#import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import argparse

plt.rcParams.update({'figure.max_open_warning': 0})

#parser.add_argument('-m', '--mode', dest='operating_mode', help="which benchmark to run", default=None, type=str, required=False)

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', dest='select_mode', help="select which set of results you want to generate graphs and tables for (basic_analysis_kitti, detailed_outliers_kitti, basic_analysis_shrec)", type=str, required=True)
args = parser.parse_args()

mode = args.select_mode

if mode == "basic_analysis_kitti":
	collected_data_path = "./results_output/collected_alignment_stats/basic_analysis_kitti_out.csv"
	output_path = "./results_output/allignment_stats_analysis_results/basic_analysis_kitti/"
elif mode == "detailed_outliers_kitti":
	collected_data_path = "./results_output/collected_alignment_stats/detailed_outliers_kitti_out.csv"
	output_path = "./results_output/allignment_stats_analysis_results/detailed_outliers_kitti/"
elif mode == "basic_analysis_shrec":
	collected_data_path = "./results_output/collected_alignment_stats/basic_analysis_shrec_out.csv"
	output_path = "./results_output/allignment_stats_analysis_results/basic_analysis_shrec/"
else:
	assert 0, "Unknown set of results: {} (allowed: basic_analysis_kitti, detailed_outliers_kitti, basic_analysis_shrec)".format(select_mode)

if mode == "detailed_outliers_kitti":
	IS_DETAILED_OUTLIERS = True
else:
	IS_DETAILED_OUTLIERS = False

if mode == "basic_analysis_shrec":
	IS_KITTI = False
else:
	IS_KITTI = True


# read in full appended csv file
stats_dataframe = pd.read_csv(
	collected_data_path, sep=",", header=None, index_col=False,
	names=[
		"seq_number","pcent_outliers","noise_stddev",
		"pcent_match_min","pcent_match_max","pcent_match_avg","pcent_match_std","pcent_match_med",
		"inlier_frob_min","inlier_frob_max","inlier_frob_avg","inlier_frob_std","inlier_frob_med",
		"fullset_frob_min","fullset_frob_max","fullset_frob_avg","fullset_frob_std","fullset_frob_med",
		"matrix_frob_min","matrix_frob_max","matrix_frob_avg","matrix_frob_std","matrix_frob_med",
		"angle_diff_min","angle_diff_max","angle_diff_avg","angle_diff_std","angle_diff_med"
	], dtype={
		"seq_number":"int","pcent_outliers":"float64","noise_stddev":"float64",
		"pcent_match_min":"float64","pcent_match_max":"float64","pcent_match_avg":"float64","pcent_match_std":"float64","pcent_match_med":"float64",
		"inlier_frob_min":"float64","inlier_frob_max":"float64","inlier_frob_avg":"float64","inlier_frob_std":"float64","inlier_frob_med":"float64",
		"fullset_frob_min":"float64","fullset_frob_max":"float64","fullset_frob_avg":"float64","fullset_frob_std":"float64","fullset_frob_med":"float64",
		"matrix_frob_min":"float64","matrix_frob_max":"float64","matrix_frob_avg":"float64","matrix_frob_std":"float64","matrix_frob_med":"float64",
		"angle_diff_min":"float64","angle_diff_max":"float64","angle_diff_avg":"float64","angle_diff_std":"float64","angle_diff_med":"float64"
	}
)

# read ranges columns to get full ranges

sequence_numbers = np.unique(stats_dataframe["seq_number"].to_numpy())
unique_outlier_settings = np.unique(stats_dataframe["pcent_outliers"].to_numpy()) # np.unique -> sorted uniwue elements
unique_stddev_settings = np.unique(stats_dataframe["noise_stddev"].to_numpy())

print("sequence numbers: {}".format(sequence_numbers))
print("outlier settings: {}".format(unique_outlier_settings))
print("noise settings: {}".format(unique_stddev_settings))

# build stats grids(s), populate, check for None
	# build dictionary with keys for each stat above
	# for each key, np.full() a table that is outlier x noise, all none
	# read through table(s) and populate from CSV
	# check for remaining none values (or rather trip over the missing row from the CSV data and error out? (rather than read csv data and populate it?)


stat_dicts = [
	dict.fromkeys([
		"pcent_match_min","pcent_match_max","pcent_match_avg","pcent_match_std","pcent_match_med",
		"inlier_frob_min","inlier_frob_max","inlier_frob_avg","inlier_frob_std","inlier_frob_med",
		"fullset_frob_min","fullset_frob_max","fullset_frob_avg","fullset_frob_std","fullset_frob_med",
		"matrix_frob_min","matrix_frob_max","matrix_frob_avg","matrix_frob_std","matrix_frob_med",
		"angle_diff_min","angle_diff_max","angle_diff_avg","angle_diff_std","angle_diff_med"
	]) for i in range(sequence_numbers.shape[0])
]

for i in range(len(stat_dicts)):
	for key in stat_dicts[i]:
		stat_dicts[i][key] = np.full(shape=(unique_outlier_settings.shape[0],unique_stddev_settings.shape[0]), fill_value=None)

# populate table(s)
# iterate over data collected

for i in range(len(stats_dataframe)):
	# for each row, consituting one set of conditions and a series of resulting measures
	
	# which sequence was this done on?
	sequence_val = stats_dataframe["seq_number"].iloc[i]
	sequence_index = np.where(sequence_numbers == sequence_val)[0][0]
	
	# get the position of the conditions, which is where to put the resulting measures
	outlier_val = stats_dataframe["pcent_outliers"].iloc[i]
	stddev_val = stats_dataframe["noise_stddev"].iloc[i]
	outlier_idx = np.flatnonzero(unique_outlier_settings == outlier_val)[0]
	stddev_idx = np.flatnonzero(unique_stddev_settings == stddev_val)[0]
	
	# for each of the kinds of measures
	for key in stat_dicts[sequence_index]:
		
		# check if there's already a value there, throw an error if there is (duplicate run)
		if stat_dicts[sequence_index][key][outlier_idx,stddev_idx] is not None:
			assert 0, " the measure {} for seq {} and outlier {}/stddev {} is not None, so a duplicate must have previously been already recorded. better stop now.".format(
				key,
				sequence_val,
				outlier_val,
				stddev_val
			)
		
		# find the right column in the data for this dict key, get the value at the current row
		# then put it in the appropriate position in the appropriate array of the dict
		stat_dicts[sequence_index][key][outlier_idx,stddev_idx] = stats_dataframe[key].iloc[i]

# check all arrays in the dict for any remaining None values (missing run)
for i in range(len(stat_dicts)):
	for key in stat_dicts[i]:
		height = stat_dicts[i][key].shape[0]
		width = stat_dicts[i][key].shape[1]
		for j in range(height):
			for k in range(width):
				if stat_dicts[i][key][j,k] is None:
					assert 0, "the element for seq {} at measure {} with conditions outlier {}/stddev {} is None, and therefore is missing. better stop now".format(
						sequence_numbers[i],
						key,
						unique_outlier_settings[j],
						unique_stddev_settings[k]
					)

# now that we know the data is complete, convert from a numpy array of object (due to None) to one of np.float64 for compatibility with matplotlib etc
# converts None to NaN, array time from object to np.float64
for i in range(len(stat_dicts)):
	for key in stat_dicts[i]:
		stat_dicts[i][key] = stat_dicts[i][key].astype(np.float64)


'''
for i in range(len(stat_dicts)):
	for key in stat_dicts[i]:
		print(key)
		print(stat_dicts[i][key])
'''

# the techniques of measurement are:
#	-percent of the landmarks that matched
#	-frob norm when comparing inlier alignment		(probably less representative, as inliers may align but not be correct)
#	-frob norm when comparing all-points alignment

#(for percent matched, more is better)
# out of min/max/avg/std/med we want:
# *** avg probably gives a decent picture of how it's doing overall
#	-std paired with avg gives some idea of spread
# *** med is probably about the same
# min probably catches the bad cases, might be informative, but they're probably outliers ---> indicates current best performance
# max probably catches the good cases, but they're probably outliers ---> indicates current bottom performance

#(for frob, less is better)
# out of min/max/avg/std/med we want:
# -min might be a good indication of the best-case matching that can be acheived ---> indicates current best performance
# -avg is probably a good "general case", perhaps
#	-std gives spread when paired with avg
# -med might be more robust to outliers than avg
# -max is likley to catch unaligned outliers
				
stat_measure_keys = stat_dicts[0].keys()

'''
"pcent_match_min","pcent_match_max","pcent_match_avg","pcent_match_std","pcent_match_med",
"inlier_frob_min","inlier_frob_max","inlier_frob_avg","inlier_frob_std","inlier_frob_med",
"fullset_frob_min","fullset_frob_max","fullset_frob_avg","fullset_frob_std","fullset_frob_med",
"matrix_frob_min","matrix_frob_max","matrix_frob_avg","matrix_frob_std","matrix_frob_med",
"angle_diff_min","angle_diff_max","angle_diff_avg","angle_diff_std","angle_diff_med"
'''

# generate and save heatmaps
for i in range(len(stat_measure_keys)):
	# generate a plot (#TODO with subplots for each sequence) for each kind of statistical measure, across the analysis approaches
	curr_key = list(stat_measure_keys)[i]
	
	#TODO temp whitelisting for testing purposes
	#if curr_key not in ["pcent_match_avg"]:
	#if curr_key not in ["fullset_frob_min"]:
	#	continue
	
	
	
	if curr_key in ["pcent_match_min","pcent_match_max","pcent_match_avg","pcent_match_std","pcent_match_med"]:
		intermediate_folder = "pcent_match"
	elif curr_key in ["inlier_frob_min","inlier_frob_max","inlier_frob_avg","inlier_frob_std","inlier_frob_med"]:
		intermediate_folder = "inlier_frob"
	elif curr_key in ["fullset_frob_min","fullset_frob_max","fullset_frob_avg","fullset_frob_std","fullset_frob_med"]:
		intermediate_folder = "fullset_frob"
	elif curr_key in ["matrix_frob_min","matrix_frob_max","matrix_frob_avg","matrix_frob_std","matrix_frob_med"]:
		intermediate_folder = "matrix_frob"
	elif curr_key in ["angle_diff_min","angle_diff_max","angle_diff_avg","angle_diff_std","angle_diff_med"]:
		intermediate_folder = "angle_difference"
	else:
		assert 0, "unknown measure: {}".format(curr_key)
	
	
	'''
	#figpath_00 = "{}/{}/{}_00.png".format(output_path,intermediate_folder,curr_key)
	#figpath_02 = "{}/{}/{}_02.png".format(output_path,intermediate_folder,curr_key)
	#figpath_05 = "{}/{}/{}_05.png".format(output_path,intermediate_folder,curr_key)
	#figpath_06 = "{}/{}/{}_06.png".format(output_path,intermediate_folder,curr_key)
	figpath_00 = "{}/{}_00.png".format(output_path,curr_key)
	figpath_02 = "{}/{}_02.png".format(output_path,curr_key)
	figpath_05 = "{}/{}_05.png".format(output_path,curr_key)
	figpath_06 = "{}/{}_06.png".format(output_path,curr_key)
	
	if not os.path.exists(os.path.dirname(figpath_00)):
		os.makedirs(os.path.dirname(figpath_00))
	if not os.path.exists(os.path.dirname(figpath_02)):
		os.makedirs(os.path.dirname(figpath_02))
	if not os.path.exists(os.path.dirname(figpath_05)):
		os.makedirs(os.path.dirname(figpath_05))
	if not os.path.exists(os.path.dirname(figpath_06)):
		os.makedirs(os.path.dirname(figpath_06))
	'''
	
	#fig_save_path = "{}/{}/{}.png".format(output_path,intermediate_folder,curr_key)
	#if not os.path.exists(os.path.dirname(fig_save_path)):
	#	os.makedirs(os.path.dirname(fig_save_path))
	
	fig_save_path = "{}/{}".format(output_path,intermediate_folder)
	if not os.path.exists(fig_save_path):
		os.makedirs(fig_save_path)
	grid_fig_save_path = "{}/{}.png".format(fig_save_path,curr_key)
	linegraph_fig_save_path = "{}/{}_lineplot.png".format(fig_save_path,curr_key)
	
	grid_table_save_path = "{}/{}_grid_tables.txt".format(fig_save_path,curr_key)
	lineplot_table_save_path = "{}/{}_lineplot_tables.txt".format(fig_save_path,curr_key)
	
	#-----------------------------------------------
	
	'''
	# subplot for seq 00
	sequence_num = sequence_numbers[0] # 00 is 0th
	data_grid = stat_dicts[sequence_num][curr_key] # returns num_outliers by num_stddevs grid for printing
	#plt.imshow(data_grid)
	#plt.figure(figsize=[12.8,9.6]) # default [6.4, 4.8]
	fig, ax = plt.subplots()
	fig.set_size_inches([12.8,9.6])
	im = ax.imshow(data_grid)
	ax.set_xticks(np.arange(len(unique_stddev_settings)), labels=unique_stddev_settings)
	ax.set_yticks(np.arange(len(unique_outlier_settings)), labels=unique_outlier_settings)
	ax.set_xlabel('Landmark Position Noise Standard Deviation (meters)')
	ax.set_ylabel('Landmark Set Outlier Percentage')
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
	# Loop over data dimensions and create text annotations.
	for i in range(len(unique_outlier_settings)):
		for j in range(len(unique_stddev_settings)):
		    text = ax.text(j,i,"{:.3f}".format(data_grid[i,j]),ha="center",va="center",color="w")
	plt.title(curr_key)
	if not os.path.exists("/tmp/testplots/"):
		os.makedirs("/tmp/testplots/")
	plt.savefig("/tmp/testplots/testimage.png")
	#plt.show()
	'''
	
	#TODO
	# we're going to ignore using sequence_numbers[] which is usually [0 2 5 6]
	# it's easier to just hardcode which plot goes where and get which sequence's data
	data_grid00 = stat_dicts[0][curr_key] # 0 -> sequence 00 typically
	'''
	if IS_KITTI:
		data_grid02 = stat_dicts[1][curr_key] # 1 -> sequence 02 typically
		data_grid05 = stat_dicts[2][curr_key] # 2 -> sequence 05 typically
		data_grid06 = stat_dicts[3][curr_key] # 3 -> sequence 06 typically
	'''
	data_grid02 = stat_dicts[1][curr_key] # 1 -> sequence 02 typically
	data_grid05 = stat_dicts[2][curr_key] # 2 -> sequence 05 typically
	data_grid06 = stat_dicts[3][curr_key] # 3 -> sequence 06 typically
	
	# experimental first-line-masking hack
	#data_grid00[0,:] = data_grid00[1,:]
	#data_grid02[0,:] = data_grid02[1,:]
	#data_grid05[0,:] = data_grid05[1,:]
	#data_grid06[0,:] = data_grid06[1,:]
	#-----------------------------------------

	#	names=[
	#	"seq_number","pcent_outliers","noise_stddev",
	#	"pcent_match_min","pcent_match_max","pcent_match_avg","pcent_match_std","pcent_match_med",
	#	"inlier_frob_min","inlier_frob_max","inlier_frob_avg","inlier_frob_std","inlier_frob_med",
	#	"fullset_frob_min","fullset_frob_max","fullset_frob_avg","fullset_frob_std","fullset_frob_med",
	#	"matrix_frob_min","matrix_frob_max","matrix_frob_avg","matrix_frob_std","matrix_frob_med",
	#	"angle_diff_min","angle_diff_max","angle_diff_avg","angle_diff_std","angle_diff_med"
	
	# pcent_match_avg, matrix_frob_avg, matrix_frob_med, angle_diff_avg
	title_mapping_dict = {
		"pcent_match_avg": "Average Percent Inliers",
		"matrix_frob_avg": "Average Frobenius Norm Error",
		"matrix_frob_med": "Median Frobenius Norm Error",
		"angle_diff_avg": "Average Angular Error"
	}
	yaxis_mapping_dict = {
		"pcent_match_avg": "Average Percentage of Landmarks Associated",
		"matrix_frob_avg": "Average Frobenius Norm Error of Estimated Transformation Matrix",
		"matrix_frob_med": "Median Frobenius Norm Error of Estimated Transformation Matrix",
		"angle_diff_avg": "Average Angular Error of Transformation (degrees)"
	}

	if curr_key in title_mapping_dict.keys():
	  title_name = title_mapping_dict[curr_key]
	  yaxis_label = yaxis_mapping_dict[curr_key]
	else:
	  title_name = curr_key
	  yaxis_label = curr_key

	#----------------------------------------------------------------------------------------------------------
	# grid of measures against outliers and stddev of noise
	fig, axes = plt.subplots(2,2,sharex=True,sharey=True, facecolor='white')#, constrained_layout=True) #orange
	#fig.set_size_inches([12.8,6.8])
	#fig.set_size_inches([25.6,11.0]) #pre-scale-upping
	#fig.set_size_inches([28.0,14.0])
	fig.set_size_inches([32.0,16.0])
	fig.suptitle("Impact of Outliers vs. Position Noise on Association ({})".format(title_name), fontsize = 30)
	fig.subplots_adjust(hspace=0.3)
	im = axes[0,0].imshow(data_grid00)
	im = axes[0,1].imshow(data_grid02)
	im = axes[1,0].imshow(data_grid05)
	im = axes[1,1].imshow(data_grid06)
	if IS_KITTI:
		axes[0,0].title.set_text("Sequence 00")
		axes[0,1].title.set_text("Sequence 02")
		axes[1,0].title.set_text("Sequence 05")
		axes[1,1].title.set_text("Sequence 06")
	else:
		axes[0,0].title.set_text("Run 1")
		axes[0,1].title.set_text("Run 2")
		axes[1,0].title.set_text("Run 3")
		axes[1,1].title.set_text("Run 4")
	''' # this would be the proper fix, oh well, keep old version for consitency with thesis plots. would also have to apply a similar fix below
	# in the case of SHREC data, there's only one sequence, "00"
	# we technically generate 4 overlaid plots like for the 4 sequences of kitti
	# but we only plot and draw a ledgend once, for the once sequence of SHREC pointclouds
	im = axes[0,0].imshow(data_grid00)
	if IS_KITTI:
		im = axes[0,1].imshow(data_grid02)
		im = axes[1,0].imshow(data_grid05)
		im = axes[1,1].imshow(data_grid06)
	if IS_KITTI:
		axes[0,0].title.set_text("Sequence 00")
		axes[0,1].title.set_text("Sequence 02")
		axes[1,0].title.set_text("Sequence 05")
		axes[1,1].title.set_text("Sequence 06")
	else:
		axes[0,0].title.set_text("Data")
	'''
	if IS_DETAILED_OUTLIERS:
		axes[0,0].set_aspect(0.727272) #other one (4) is 8 tall, this one is 11. everything is sized for 4, so squish this thing by setting aspect to 8/11=0.727272 to make it work out
		axes[0,1].set_aspect(0.727272)
		axes[1,0].set_aspect(0.727272)
		axes[1,1].set_aspect(0.727272)
	# axis (including labels) are set as shared above when defining number of subplots
	axes[0,0].set_xticks(np.arange(len(unique_stddev_settings)), labels=unique_stddev_settings)
	if IS_DETAILED_OUTLIERS:
		axes[0,0].set_yticks(np.arange(len(unique_outlier_settings)), labels=["0/120","1/120","2/120","3/120","4/120","5/120","6/120","7/120","8/120","9/120","10/120"])
	else:
		axes[0,0].set_yticks(np.arange(len(unique_outlier_settings)), labels=unique_outlier_settings)
	# only actually needed on outer plots, but whatever
	plt.setp(axes[0,0].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
	plt.setp(axes[0,1].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
	plt.setp(axes[1,0].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
	plt.setp(axes[1,1].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
	# Loop over data dimensions and create text annotations.
	for i in range(len(unique_outlier_settings)):
		for j in range(len(unique_stddev_settings)):
			text = axes[0,0].text(j,i,"{:.3f}".format(data_grid00[i,j]),ha="center",va="center",color="w")
			text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='black')])
			text = axes[0,1].text(j,i,"{:.3f}".format(data_grid02[i,j]),ha="center",va="center",color="w")
			text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='black')])
			text = axes[1,0].text(j,i,"{:.3f}".format(data_grid05[i,j]),ha="center",va="center",color="w")
			text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='black')])
			text = axes[1,1].text(j,i,"{:.3f}".format(data_grid06[i,j]),ha="center",va="center",color="w")
			text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='black')])
	#fig.subplots_adjust(hspace=3.0) # doesn't seem to do anything
	                                                                                                                      #TODO needs to change based on current measure
	# create whole-plot axis labels using big invisible subplot
	invis_ax = fig.add_subplot(111, frameon=False) # add a big axes, hide frame
	plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False) # hide tick and tick label of the big axes
	plt.grid(False)
	plt.xlabel("Landmark Position Noise Standard Deviation (meters)", fontsize = 20, labelpad=20)
	plt.ylabel("Landmark Set Outlier Percentage", fontsize = 20, labelpad=20)
	invis_ax.xaxis.set_label_coords(0.5, 0.02)
	fig.tight_layout(h_pad=3.0)
	#plt.show()
	plt.savefig(grid_fig_save_path)
	
	def round_sigfig(number,figs):
		return '%s' % float('%.{}g'.format(figs) % number)
	
	if curr_key == "pcent_match_avg":
		round_digits = 2
	elif curr_key == "matrix_frob_med":
		round_digits = 2
	elif curr_key == "angle_diff_avg":
		round_digits = 3
	else:
		round_digits = 2
	
	#TODO
	grid_table_textfile = open(grid_table_save_path, "w")
	print("\\begin{table}"																																		, file=grid_table_textfile)
	#print("\\centering"																																		, file=grid_table_textfile)
	print("\\makebox[\\textwidth][c]{"																															, file=grid_table_textfile)
	print("\\footnotesize"																																		, file=grid_table_textfile)
	print("\\setlength{\\tabcolsep}{3pt}"																														, file=grid_table_textfile)
	print("\\begin{tabular}{|| c || " + " | ".join(["c" for i in range(len(unique_stddev_settings))]) + "||}"													, file=grid_table_textfile)
	#------------------------------------------
	print("\\hline"																																				, file=grid_table_textfile)
	print("\\multicolumn{22}{||c||}{Sensitivity to Noise vs Outliers: Sequence 00 } \\\\ [0.5ex]"																, file=grid_table_textfile)
	print("\\hline"																																				, file=grid_table_textfile)
	print("& \\multicolumn{21}{|c||}{Noise Standard Deviation [m]} \\\\ [0.5ex]"																				, file=grid_table_textfile)
	print("\\hline"																																				, file=grid_table_textfile)
	print("\\% Outliers & " + " & ".join([str(round_sigfig(x,round_digits)) for x in unique_stddev_settings]) + " \\\\"															, file=grid_table_textfile)
	print("\\hline\\hline"																																		, file=grid_table_textfile)
	for i in range(len(unique_outlier_settings)):
		print(str(100*unique_outlier_settings[i]) + " & " + " & ".join([str(round_sigfig(x,round_digits)) for x in data_grid00[i,:]]) + " \\\\"									, file=grid_table_textfile)
		print("\\hline"																																			, file=grid_table_textfile)
	#print("\\hline"																																			, file=grid_table_textfile)
	#------------------------------------------
	print("\\hline"																																				, file=grid_table_textfile)
	print("\\multicolumn{22}{||c||}{Sensitivity to Noise vs Outliers: Sequence 02 } \\\\ [0.5ex]"																, file=grid_table_textfile)
	print("\\hline"																																				, file=grid_table_textfile)
	print("& \\multicolumn{21}{|c||}{Noise Standard Deviation [m]} \\\\ [0.5ex]"																				, file=grid_table_textfile)
	print("\\hline"																																				, file=grid_table_textfile)
	print("\\% Outliers & " + " & ".join([str(round_sigfig(x,round_digits)) for x in unique_stddev_settings]) + " \\\\"															, file=grid_table_textfile)
	print("\\hline\\hline"																																		, file=grid_table_textfile)
	for i in range(len(unique_outlier_settings)):
		print(str(100*unique_outlier_settings[i]) + " & " + " & ".join([str(round_sigfig(x,round_digits)) for x in data_grid02[i,:]]) + " \\\\"									, file=grid_table_textfile)
		print("\\hline"																																			, file=grid_table_textfile)
	#print("\\hline"																																			, file=grid_table_textfile)
	#------------------------------------------
	print("\\hline"																																				, file=grid_table_textfile)
	print("\\multicolumn{22}{||c||}{Sensitivity to Noise vs Outliers: Sequence 05 } \\\\ [0.5ex]"																, file=grid_table_textfile)
	print("\\hline"																																				, file=grid_table_textfile)
	print("& \\multicolumn{21}{|c||}{Noise Standard Deviation [m]} \\\\ [0.5ex]"																				, file=grid_table_textfile)
	print("\\hline"																																				, file=grid_table_textfile)
	print("\\% Outliers & " + " & ".join([str(round_sigfig(x,round_digits)) for x in unique_stddev_settings]) + " \\\\"															, file=grid_table_textfile)
	print("\\hline\\hline"																																		, file=grid_table_textfile)
	for i in range(len(unique_outlier_settings)):
		print(str(100*unique_outlier_settings[i]) + " & " + " & ".join([str(round_sigfig(x,round_digits)) for x in data_grid05[i,:]]) + " \\\\"									, file=grid_table_textfile)
		print("\\hline"																																			, file=grid_table_textfile)
	#print("\\hline"																																			, file=grid_table_textfile)
	#------------------------------------------
	print("\\hline"																																				, file=grid_table_textfile)
	print("\\multicolumn{22}{||c||}{Sensitivity to Noise vs Outliers: Sequence 06 } \\\\ [0.5ex]"																, file=grid_table_textfile)
	print("\\hline"																																				, file=grid_table_textfile)
	print("& \\multicolumn{21}{|c||}{Noise Standard Deviation [m]} \\\\ [0.5ex]"																				, file=grid_table_textfile)
	print("\\hline"																																				, file=grid_table_textfile)
	print("\\% Outliers & " + " & ".join([str(round_sigfig(x,round_digits)) for x in unique_stddev_settings]) + " \\\\"															, file=grid_table_textfile)
	print("\\hline\\hline"																																		, file=grid_table_textfile)
	for i in range(len(unique_outlier_settings)):
		print(str(100*unique_outlier_settings[i]) + " & " + " & ".join([str(round_sigfig(x,round_digits)) for x in data_grid06[i,:]]) + " \\\\"									, file=grid_table_textfile)
		print("\\hline"																																			, file=grid_table_textfile)
	#print("\\hline"																																			, file=grid_table_textfile)
	#------------------------------------------
	print("\\end{tabular}"																																		, file=grid_table_textfile)
	print("}"																																					, file=grid_table_textfile)
	print("\\caption[FIXME " + curr_key.replace("_","") + " outliers vs stddev short caption.]{FIXME " + curr_key.replace("_","\_") + " outliers vs stddev long caption.}"		, file=grid_table_textfile)
	print("\\label{tab:cello}"																																	, file=grid_table_textfile)
	print("\\end{table}"																																		, file=grid_table_textfile)
	grid_table_textfile.close()
	
	#----------------------------------------------------------------------------------------------------------
	
	fig,(ax1, ax2) = plt.subplots(2,1, facecolor='white') #yellow
	fig.set_size_inches([12.8,7.2])
	fig.subplots_adjust(hspace=0.5)
	#fig.tight_layout()
	fig.suptitle("Impact of Outliers and Position Noise on Association, Independently ({})".format(title_name), fontsize = 15)

	# slice of measures grid (against outliers)
	ax1.plot(unique_outlier_settings,data_grid00[:,0], marker='.')
	ax1.plot(unique_outlier_settings,data_grid02[:,0], marker='.')
	ax1.plot(unique_outlier_settings,data_grid05[:,0], marker='.')
	ax1.plot(unique_outlier_settings,data_grid06[:,0], marker='.')
	if IS_KITTI:
		ax1.legend(["Sequence 00","Sequence 02","sequence 05","Sequence 06"], loc ="upper right")
	else:
		ax1.legend(["Run 1","Run 2","Run 3","Run 4"], loc ="upper right")
	plt.setp(ax1.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
	ax1.locator_params(axis='y', nbins=12)
	#select one or the other depending on which plot this is
	if IS_DETAILED_OUTLIERS:
		ax1.set_xticks(unique_outlier_settings, ["0/120","1/120","2/120","3/120","4/120","5/120","6/120","7/120","8/120","9/120","10/120"])
	else:
		ax1.set_xticks(unique_outlier_settings)
	ax1.set(xlabel="Percentage of Outlier Landmarks")

	# slice of measures grid (against stddev of noise)
	ax2.plot(unique_stddev_settings,data_grid00[0,:], marker='.')
	ax2.plot(unique_stddev_settings,data_grid02[0,:], marker='.')
	ax2.plot(unique_stddev_settings,data_grid05[0,:], marker='.')
	ax2.plot(unique_stddev_settings,data_grid06[0,:], marker='.')
	if IS_KITTI:
		ax2.legend(["Sequence 00","Sequence 02","sequence 05","Sequence 06"], loc ="upper right")
	else:
		ax2.legend(["Run 1","Run 2","Run 3","Run 4"], loc ="upper right")
	plt.setp(ax2.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
	ax2.locator_params(axis='y', nbins=12)
	ax2.set_xticks(unique_stddev_settings)
	ax2.set(xlabel="Landmark Position Noise Stddev [m]")

	# create whole-plot axis labels using big invisible subplot
	fig.add_subplot(111, frameon=False) # add a big axes, hide frame
	plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False) # hide tick and tick label of the big axes
	plt.grid(False)
	plt.ylabel(yaxis_label, fontsize = 10, labelpad=20)
	plt.savefig(linegraph_fig_save_path)
	
	lineplot_table_textfile = open(lineplot_table_save_path, "w")
	print("\\begin{table}"																											, file=lineplot_table_textfile)
	print("\\centering"																												, file=lineplot_table_textfile)
	print("\\begin{tabular}{|| c || " + " ".join(["c" for i in range(len(unique_outlier_settings))]) + "||}"						, file=lineplot_table_textfile)
	print("\\hline"																													, file=lineplot_table_textfile)
	print("Outlier Percent & " + " & ".join([str(100*x) for x in unique_outlier_settings]) + " \\\\ [0.5ex]"						, file=lineplot_table_textfile)
	print("\\hline\\hline"																											, file=lineplot_table_textfile)
	print("Seq 00 " + curr_key.replace("_","") + " & " + " & ".join([str(round(x,2)) for x in data_grid00[:,0]]) + " \\\\"			, file=lineplot_table_textfile)
	print("\\hline"																													, file=lineplot_table_textfile)
	print("Seq 02 " + curr_key.replace("_","") + " & " + " & ".join([str(round(x,2)) for x in data_grid02[:,0]]) + " \\\\"			, file=lineplot_table_textfile)
	print("\\hline"																													, file=lineplot_table_textfile)
	print("Seq 05 " + curr_key.replace("_","") + " & " + " & ".join([str(round(x,2)) for x in data_grid05[:,0]]) + " \\\\"			, file=lineplot_table_textfile)
	print("\\hline"																													, file=lineplot_table_textfile)
	print("Seq 06 " + curr_key.replace("_","") + " & " + " & ".join([str(round(x,2)) for x in data_grid06[:,0]]) + " \\\\"			, file=lineplot_table_textfile)
	print("\\hline"																													, file=lineplot_table_textfile)
	print("\\end{tabular}"																											, file=lineplot_table_textfile)
	print("\\caption[FIXME " + curr_key.replace("_","") + " vs outliers short caption.]{FIXME " + curr_key.replace("_","\_") + "vs outliers long caption.}", file=lineplot_table_textfile)
	print("\\label{tab:cello}"																										, file=lineplot_table_textfile)
	print("\\end{table}"																											, file=lineplot_table_textfile)
	
	print(""																														, file=lineplot_table_textfile)
	print(""																														, file=lineplot_table_textfile)
	print(""																														, file=lineplot_table_textfile)

	print("\\begin{table}"																											, file=lineplot_table_textfile)
	print("\\centering"																												, file=lineplot_table_textfile)
	print("\\begin{tabular}{|| c || " + " ".join(["c" for i in range(len(unique_stddev_settings))]) + "||}"							, file=lineplot_table_textfile)
	print("\\hline"																													, file=lineplot_table_textfile)
	print("Noise Std Dev & " + " & ".join([str(round(x,2)) for x in unique_stddev_settings]) + " \\\\ [0.5ex]"						, file=lineplot_table_textfile)
	print("\\hline\\hline"																											, file=lineplot_table_textfile)
	print("Seq 00 " + curr_key.replace("_","") + " & " + " & ".join([str(round(x,2)) for x in data_grid00[0,:]]) + " \\\\"			, file=lineplot_table_textfile)
	print("\\hline"																													, file=lineplot_table_textfile)
	print("Seq 02 " + curr_key.replace("_","") + " & " + " & ".join([str(round(x,2)) for x in data_grid02[0,:]]) + " \\\\"			, file=lineplot_table_textfile)
	print("\\hline"																													, file=lineplot_table_textfile)
	print("Seq 05 " + curr_key.replace("_","") + " & " + " & ".join([str(round(x,2)) for x in data_grid05[0,:]]) + " \\\\"			, file=lineplot_table_textfile)
	print("\\hline"																													, file=lineplot_table_textfile)
	print("Seq 06 " + curr_key.replace("_","") + " & " + " & ".join([str(round(x,2)) for x in data_grid06[0,:]]) + " \\\\"			, file=lineplot_table_textfile)
	print("\\hline"																													, file=lineplot_table_textfile)
	print("\\end{tabular}"																											, file=lineplot_table_textfile)
	print("\\caption[FIXME " + curr_key.replace("_","") + " vs stddev short caption.]{FIXME " + curr_key.replace("_","\_") + "vs stddev long caption.}", file=lineplot_table_textfile)
	print("\\label{tab:cello}"																										, file=lineplot_table_textfile)
	print("\\end{table}"																											, file=lineplot_table_textfile)
	lineplot_table_textfile.close()
	
	pass

print("output_path: {}".format(output_path))

'''
# generate and save heatmaps
for i in range(len(stat_dicts)):
	sequence_num = sequence_numbers[i]
	
	for key in stat_dicts[i]:
		height = stat_dicts[i][key].shape[0]
		width = stat_dicts[i][key].shape[1]
		for j in range(height):
			for k in range(width):
				
				
				
				pass
'''


# generate and save tables





















