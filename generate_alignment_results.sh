#!/bin/bash
set -e

# optional check for all the output files that should exist given a complete set of results with (a default of) 8 concurrent threads
: '
test_file_exists () {
	if [ ! -f $1 ]; then
		echo "$1 not found!"
		exit 1
	fi
}

test_file_exists ./results_output/collected_alignment_stats/basic_analysis_kitti_1/append_collected_stats.csv
test_file_exists ./results_output/collected_alignment_stats/basic_analysis_kitti_2/append_collected_stats.csv
test_file_exists ./results_output/collected_alignment_stats/basic_analysis_kitti_0/append_collected_stats.csv
test_file_exists ./results_output/collected_alignment_stats/basic_analysis_kitti_6/append_collected_stats.csv
test_file_exists ./results_output/collected_alignment_stats/basic_analysis_kitti_5/append_collected_stats.csv
test_file_exists ./results_output/collected_alignment_stats/basic_analysis_kitti_7/append_collected_stats.csv
test_file_exists ./results_output/collected_alignment_stats/basic_analysis_kitti_4/append_collected_stats.csv
test_file_exists ./results_output/collected_alignment_stats/basic_analysis_kitti_3/append_collected_stats.csv

test_file_exists ./results_output/collected_alignment_stats/basic_analysis_shrec_0/append_collected_stats.csv
test_file_exists ./results_output/collected_alignment_stats/basic_analysis_shrec_2/append_collected_stats.csv
test_file_exists ./results_output/collected_alignment_stats/basic_analysis_shrec_4/append_collected_stats.csv
test_file_exists ./results_output/collected_alignment_stats/basic_analysis_shrec_1/append_collected_stats.csv
test_file_exists ./results_output/collected_alignment_stats/basic_analysis_shrec_3/append_collected_stats.csv
test_file_exists ./results_output/collected_alignment_stats/basic_analysis_shrec_6/append_collected_stats.csv
test_file_exists ./results_output/collected_alignment_stats/basic_analysis_shrec_5/append_collected_stats.csv
test_file_exists ./results_output/collected_alignment_stats/basic_analysis_shrec_7/append_collected_stats.csv

test_file_exists ./results_output/collected_alignment_stats/detailed_outliers_kitti_4/append_collected_stats.csv
test_file_exists ./results_output/collected_alignment_stats/detailed_outliers_kitti_0/append_collected_stats.csv
test_file_exists ./results_output/collected_alignment_stats/detailed_outliers_kitti_2/append_collected_stats.csv
test_file_exists ./results_output/collected_alignment_stats/detailed_outliers_kitti_5/append_collected_stats.csv
test_file_exists ./results_output/collected_alignment_stats/detailed_outliers_kitti_7/append_collected_stats.csv
test_file_exists ./results_output/collected_alignment_stats/detailed_outliers_kitti_1/append_collected_stats.csv
test_file_exists ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/append_collected_stats.csv
test_file_exists ./results_output/collected_alignment_stats/detailed_outliers_kitti_6/append_collected_stats.csv
 # '

#find ./results_output/collected_alignment_stats/ | grep basic_analysis_kitti | grep \\.csv | grep -v _out\\.
#find ./results_output/collected_alignment_stats/ | grep basic_analysis_shrec | grep \\.csv | grep -v _out\\.
#find ./results_output/collected_alignment_stats/ | grep detailed_outliers_kitti | grep \\.csv | grep -v _out\\.

# collect the results of all the different threads together
rm -f ./results_output/collected_alignment_stats/basic_analysis_kitti_out.csv
rm -f ./results_output/collected_alignment_stats/detailed_outliers_kitti_out.csv
rm -f ./results_output/collected_alignment_stats/basic_analysis_shrec_out.csv
find ./results_output/collected_alignment_stats/ | grep basic_analysis_kitti	| grep \\.csv | grep -v _out\\. | awk '{system("cat "$1)}' > ./results_output/collected_alignment_stats/basic_analysis_kitti_out.csv
find ./results_output/collected_alignment_stats/ | grep detailed_outliers_kitti	| grep \\.csv | grep -v _out\\. | awk '{system("cat "$1)}' > ./results_output/collected_alignment_stats/detailed_outliers_kitti_out.csv
find ./results_output/collected_alignment_stats/ | grep basic_analysis_shrec	| grep \\.csv | grep -v _out\\. | awk '{system("cat "$1)}' > ./results_output/collected_alignment_stats/basic_analysis_shrec_out.csv





