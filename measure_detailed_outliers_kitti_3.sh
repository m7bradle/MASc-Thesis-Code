#!/bin/bash
set -e
echo "runfile: doing run for 0.025 outliers and 1.35 stddev on seq 06"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 06 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.025 --std_dev_noise 1.35 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 1 of 127 "
echo "runfile: doing run for 0.034 outliers and 0.15 stddev on seq 00"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 00 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.034 --std_dev_noise 0.15 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 2 of 127 "
echo "runfile: doing run for 0.084 outliers and 1.875 stddev on seq 02"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 02 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.084 --std_dev_noise 1.875 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 3 of 127 "
echo "runfile: doing run for 0.017 outliers and 3.75 stddev on seq 00"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 00 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.017 --std_dev_noise 3.75 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 4 of 127 "
echo "runfile: doing run for 0.075 outliers and 0.0 stddev on seq 06"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 06 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.075 --std_dev_noise 0.0 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 5 of 127 "
echo "runfile: doing run for 0.025 outliers and 1.875 stddev on seq 06"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 06 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.025 --std_dev_noise 1.875 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 6 of 127 "
echo "runfile: doing run for 0.017 outliers and 2.25 stddev on seq 06"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 06 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.017 --std_dev_noise 2.25 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 7 of 127 "
echo "runfile: doing run for 0.059 outliers and 1.875 stddev on seq 02"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 02 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.059 --std_dev_noise 1.875 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 8 of 127 "
echo "runfile: doing run for 0.0084 outliers and 0.0 stddev on seq 00"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 00 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.0084 --std_dev_noise 0.0 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 9 of 127 "
echo "runfile: doing run for 0.0084 outliers and 0.3 stddev on seq 02"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 02 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.0084 --std_dev_noise 0.3 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 10 of 127 "
echo "runfile: doing run for 0.042 outliers and 1.35 stddev on seq 05"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 05 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.042 --std_dev_noise 1.35 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 11 of 127 "
echo "runfile: doing run for 0.075 outliers and 0.45 stddev on seq 02"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 02 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.075 --std_dev_noise 0.45 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 12 of 127 "
echo "runfile: doing run for 0.025 outliers and 0.3 stddev on seq 05"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 05 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.025 --std_dev_noise 0.3 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 13 of 127 "
echo "runfile: doing run for 0.059 outliers and 2.25 stddev on seq 02"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 02 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.059 --std_dev_noise 2.25 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 14 of 127 "
echo "runfile: doing run for 0.034 outliers and 3.75 stddev on seq 00"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 00 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.034 --std_dev_noise 3.75 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 15 of 127 "
echo "runfile: doing run for 0.0 outliers and 0.6 stddev on seq 00"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 00 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.0 --std_dev_noise 0.6 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 16 of 127 "
echo "runfile: doing run for 0.025 outliers and 1.5 stddev on seq 02"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 02 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.025 --std_dev_noise 1.5 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 17 of 127 "
echo "runfile: doing run for 0.0 outliers and 4.875 stddev on seq 02"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 02 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.0 --std_dev_noise 4.875 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 18 of 127 "
echo "runfile: doing run for 0.05 outliers and 1.5 stddev on seq 00"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 00 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.05 --std_dev_noise 1.5 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 19 of 127 "
echo "runfile: doing run for 0.067 outliers and 1.05 stddev on seq 02"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 02 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.067 --std_dev_noise 1.05 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 20 of 127 "
echo "runfile: doing run for 0.0 outliers and 3.75 stddev on seq 00"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 00 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.0 --std_dev_noise 3.75 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 21 of 127 "
echo "runfile: doing run for 0.017 outliers and 0.9 stddev on seq 02"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 02 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.017 --std_dev_noise 0.9 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 22 of 127 "
echo "runfile: doing run for 0.034 outliers and 1.5 stddev on seq 00"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 00 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.034 --std_dev_noise 1.5 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 23 of 127 "
echo "runfile: doing run for 0.067 outliers and 5.25 stddev on seq 00"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 00 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.067 --std_dev_noise 5.25 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 24 of 127 "
echo "runfile: doing run for 0.067 outliers and 0.15 stddev on seq 05"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 05 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.067 --std_dev_noise 0.15 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 25 of 127 "
echo "runfile: doing run for 0.075 outliers and 3.75 stddev on seq 05"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 05 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.075 --std_dev_noise 3.75 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 26 of 127 "
echo "runfile: doing run for 0.067 outliers and 5.25 stddev on seq 02"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 02 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.067 --std_dev_noise 5.25 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 27 of 127 "
echo "runfile: doing run for 0.059 outliers and 0.75 stddev on seq 00"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 00 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.059 --std_dev_noise 0.75 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 28 of 127 "
echo "runfile: doing run for 0.084 outliers and 3.75 stddev on seq 05"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 05 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.084 --std_dev_noise 3.75 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 29 of 127 "
echo "runfile: doing run for 0.05 outliers and 6.0 stddev on seq 06"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 06 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.05 --std_dev_noise 6.0 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 30 of 127 "
echo "runfile: doing run for 0.0084 outliers and 5.625 stddev on seq 06"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 06 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.0084 --std_dev_noise 5.625 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 31 of 127 "
echo "runfile: doing run for 0.017 outliers and 5.25 stddev on seq 02"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 02 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.017 --std_dev_noise 5.25 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 32 of 127 "
echo "runfile: doing run for 0.017 outliers and 0.15 stddev on seq 02"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 02 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.017 --std_dev_noise 0.15 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 33 of 127 "
echo "runfile: doing run for 0.042 outliers and 0.6 stddev on seq 05"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 05 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.042 --std_dev_noise 0.6 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 34 of 127 "
echo "runfile: doing run for 0.0 outliers and 1.35 stddev on seq 05"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 05 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.0 --std_dev_noise 1.35 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 35 of 127 "
echo "runfile: doing run for 0.075 outliers and 1.875 stddev on seq 05"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 05 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.075 --std_dev_noise 1.875 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 36 of 127 "
echo "runfile: doing run for 0.0 outliers and 3.75 stddev on seq 05"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 05 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.0 --std_dev_noise 3.75 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 37 of 127 "
echo "runfile: doing run for 0.025 outliers and 0.0 stddev on seq 06"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 06 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.025 --std_dev_noise 0.0 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 38 of 127 "
echo "runfile: doing run for 0.017 outliers and 0.9 stddev on seq 00"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 00 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.017 --std_dev_noise 0.9 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 39 of 127 "
echo "runfile: doing run for 0.084 outliers and 1.05 stddev on seq 06"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 06 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.084 --std_dev_noise 1.05 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 40 of 127 "
echo "runfile: doing run for 0.05 outliers and 0.9 stddev on seq 02"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 02 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.05 --std_dev_noise 0.9 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 41 of 127 "
echo "runfile: doing run for 0.0 outliers and 4.125 stddev on seq 05"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 05 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.0 --std_dev_noise 4.125 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 42 of 127 "
echo "runfile: doing run for 0.017 outliers and 1.5 stddev on seq 05"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 05 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.017 --std_dev_noise 1.5 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 43 of 127 "
echo "runfile: doing run for 0.017 outliers and 1.05 stddev on seq 00"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 00 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.017 --std_dev_noise 1.05 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 44 of 127 "
echo "runfile: doing run for 0.05 outliers and 3.375 stddev on seq 05"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 05 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.05 --std_dev_noise 3.375 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 45 of 127 "
echo "runfile: doing run for 0.05 outliers and 1.2 stddev on seq 06"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 06 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.05 --std_dev_noise 1.2 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 46 of 127 "
echo "runfile: doing run for 0.084 outliers and 2.25 stddev on seq 06"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 06 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.084 --std_dev_noise 2.25 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 47 of 127 "
echo "runfile: doing run for 0.0084 outliers and 5.25 stddev on seq 02"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 02 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.0084 --std_dev_noise 5.25 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 48 of 127 "
echo "runfile: doing run for 0.034 outliers and 4.125 stddev on seq 00"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 00 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.034 --std_dev_noise 4.125 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 49 of 127 "
echo "runfile: doing run for 0.059 outliers and 1.2 stddev on seq 06"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 06 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.059 --std_dev_noise 1.2 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 50 of 127 "
echo "runfile: doing run for 0.042 outliers and 0.3 stddev on seq 00"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 00 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.042 --std_dev_noise 0.3 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 51 of 127 "
echo "runfile: doing run for 0.0 outliers and 3.0 stddev on seq 00"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 00 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.0 --std_dev_noise 3.0 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 52 of 127 "
echo "runfile: doing run for 0.059 outliers and 0.6 stddev on seq 00"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 00 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.059 --std_dev_noise 0.6 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 53 of 127 "
echo "runfile: doing run for 0.0084 outliers and 0.0 stddev on seq 02"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 02 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.0084 --std_dev_noise 0.0 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 54 of 127 "
echo "runfile: doing run for 0.05 outliers and 4.125 stddev on seq 06"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 06 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.05 --std_dev_noise 4.125 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 55 of 127 "
echo "runfile: doing run for 0.084 outliers and 4.125 stddev on seq 02"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 02 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.084 --std_dev_noise 4.125 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 56 of 127 "
echo "runfile: doing run for 0.017 outliers and 3.75 stddev on seq 02"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 02 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.017 --std_dev_noise 3.75 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 57 of 127 "
echo "runfile: doing run for 0.084 outliers and 1.5 stddev on seq 05"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 05 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.084 --std_dev_noise 1.5 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 58 of 127 "
echo "runfile: doing run for 0.059 outliers and 4.5 stddev on seq 02"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 02 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.059 --std_dev_noise 4.5 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 59 of 127 "
echo "runfile: doing run for 0.05 outliers and 4.125 stddev on seq 02"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 02 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.05 --std_dev_noise 4.125 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 60 of 127 "
echo "runfile: doing run for 0.0 outliers and 0.9 stddev on seq 06"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 06 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.0 --std_dev_noise 0.9 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 61 of 127 "
echo "runfile: doing run for 0.025 outliers and 2.625 stddev on seq 02"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 02 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.025 --std_dev_noise 2.625 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 62 of 127 "
echo "runfile: doing run for 0.059 outliers and 1.35 stddev on seq 06"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 06 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.059 --std_dev_noise 1.35 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 63 of 127 "
echo "runfile: doing run for 0.075 outliers and 2.25 stddev on seq 00"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 00 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.075 --std_dev_noise 2.25 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 64 of 127 "
echo "runfile: doing run for 0.05 outliers and 1.2 stddev on seq 02"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 02 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.05 --std_dev_noise 1.2 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 65 of 127 "
echo "runfile: doing run for 0.084 outliers and 5.25 stddev on seq 02"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 02 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.084 --std_dev_noise 5.25 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 66 of 127 "
echo "runfile: doing run for 0.017 outliers and 5.625 stddev on seq 05"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 05 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.017 --std_dev_noise 5.625 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 67 of 127 "
echo "runfile: doing run for 0.059 outliers and 2.25 stddev on seq 06"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 06 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.059 --std_dev_noise 2.25 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 68 of 127 "
echo "runfile: doing run for 0.067 outliers and 1.35 stddev on seq 06"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 06 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.067 --std_dev_noise 1.35 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 69 of 127 "
echo "runfile: doing run for 0.017 outliers and 0.0 stddev on seq 06"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 06 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.017 --std_dev_noise 0.0 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 70 of 127 "
echo "runfile: doing run for 0.042 outliers and 0.9 stddev on seq 05"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 05 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.042 --std_dev_noise 0.9 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 71 of 127 "
echo "runfile: doing run for 0.084 outliers and 1.05 stddev on seq 05"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 05 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.084 --std_dev_noise 1.05 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 72 of 127 "
echo "runfile: doing run for 0.075 outliers and 1.5 stddev on seq 06"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 06 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.075 --std_dev_noise 1.5 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 73 of 127 "
echo "runfile: doing run for 0.025 outliers and 2.25 stddev on seq 05"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 05 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.025 --std_dev_noise 2.25 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 74 of 127 "
echo "runfile: doing run for 0.0 outliers and 1.35 stddev on seq 00"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 00 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.0 --std_dev_noise 1.35 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 75 of 127 "
echo "runfile: doing run for 0.067 outliers and 3.375 stddev on seq 02"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 02 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.067 --std_dev_noise 3.375 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 76 of 127 "
echo "runfile: doing run for 0.084 outliers and 0.6 stddev on seq 05"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 05 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.084 --std_dev_noise 0.6 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 77 of 127 "
echo "runfile: doing run for 0.025 outliers and 0.15 stddev on seq 00"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 00 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.025 --std_dev_noise 0.15 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 78 of 127 "
echo "runfile: doing run for 0.0084 outliers and 2.625 stddev on seq 00"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 00 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.0084 --std_dev_noise 2.625 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 79 of 127 "
echo "runfile: doing run for 0.084 outliers and 4.5 stddev on seq 00"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 00 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.084 --std_dev_noise 4.5 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 80 of 127 "
echo "runfile: doing run for 0.059 outliers and 0.75 stddev on seq 06"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 06 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.059 --std_dev_noise 0.75 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 81 of 127 "
echo "runfile: doing run for 0.084 outliers and 0.0 stddev on seq 00"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 00 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.084 --std_dev_noise 0.0 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 82 of 127 "
echo "runfile: doing run for 0.059 outliers and 0.3 stddev on seq 02"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 02 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.059 --std_dev_noise 0.3 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 83 of 127 "
echo "runfile: doing run for 0.042 outliers and 2.625 stddev on seq 00"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 00 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.042 --std_dev_noise 2.625 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 84 of 127 "
echo "runfile: doing run for 0.067 outliers and 3.75 stddev on seq 02"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 02 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.067 --std_dev_noise 3.75 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 85 of 127 "
echo "runfile: doing run for 0.0 outliers and 0.9 stddev on seq 00"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 00 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.0 --std_dev_noise 0.9 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 86 of 127 "
echo "runfile: doing run for 0.05 outliers and 1.875 stddev on seq 00"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 00 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.05 --std_dev_noise 1.875 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 87 of 127 "
echo "runfile: doing run for 0.059 outliers and 6.0 stddev on seq 00"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 00 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.059 --std_dev_noise 6.0 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 88 of 127 "
echo "runfile: doing run for 0.075 outliers and 0.9 stddev on seq 02"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 02 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.075 --std_dev_noise 0.9 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 89 of 127 "
echo "runfile: doing run for 0.075 outliers and 4.875 stddev on seq 02"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 02 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.075 --std_dev_noise 4.875 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 90 of 127 "
echo "runfile: doing run for 0.059 outliers and 6.0 stddev on seq 06"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 06 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.059 --std_dev_noise 6.0 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 91 of 127 "
echo "runfile: doing run for 0.017 outliers and 2.25 stddev on seq 02"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 02 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.017 --std_dev_noise 2.25 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 92 of 127 "
echo "runfile: doing run for 0.025 outliers and 0.45 stddev on seq 02"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 02 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.025 --std_dev_noise 0.45 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 93 of 127 "
echo "runfile: doing run for 0.017 outliers and 1.35 stddev on seq 02"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 02 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.017 --std_dev_noise 1.35 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 94 of 127 "
echo "runfile: doing run for 0.025 outliers and 3.375 stddev on seq 05"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 05 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.025 --std_dev_noise 3.375 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 95 of 127 "
echo "runfile: doing run for 0.0 outliers and 1.875 stddev on seq 00"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 00 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.0 --std_dev_noise 1.875 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 96 of 127 "
echo "runfile: doing run for 0.025 outliers and 3.0 stddev on seq 05"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 05 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.025 --std_dev_noise 3.0 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 97 of 127 "
echo "runfile: doing run for 0.025 outliers and 1.05 stddev on seq 02"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 02 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.025 --std_dev_noise 1.05 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 98 of 127 "
echo "runfile: doing run for 0.0 outliers and 2.25 stddev on seq 02"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 02 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.0 --std_dev_noise 2.25 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 99 of 127 "
echo "runfile: doing run for 0.067 outliers and 0.9 stddev on seq 05"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 05 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.067 --std_dev_noise 0.9 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 100 of 127 "
echo "runfile: doing run for 0.042 outliers and 3.75 stddev on seq 02"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 02 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.042 --std_dev_noise 3.75 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 101 of 127 "
echo "runfile: doing run for 0.025 outliers and 0.9 stddev on seq 06"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 06 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.025 --std_dev_noise 0.9 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 102 of 127 "
echo "runfile: doing run for 0.017 outliers and 4.5 stddev on seq 00"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 00 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.017 --std_dev_noise 4.5 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 103 of 127 "
echo "runfile: doing run for 0.0084 outliers and 0.3 stddev on seq 00"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 00 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.0084 --std_dev_noise 0.3 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 104 of 127 "
echo "runfile: doing run for 0.05 outliers and 0.15 stddev on seq 06"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 06 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.05 --std_dev_noise 0.15 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 105 of 127 "
echo "runfile: doing run for 0.075 outliers and 5.25 stddev on seq 05"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 05 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.075 --std_dev_noise 5.25 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 106 of 127 "
echo "runfile: doing run for 0.0084 outliers and 3.75 stddev on seq 06"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 06 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.0084 --std_dev_noise 3.75 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 107 of 127 "
echo "runfile: doing run for 0.0 outliers and 1.05 stddev on seq 05"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 05 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.0 --std_dev_noise 1.05 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 108 of 127 "
echo "runfile: doing run for 0.025 outliers and 1.2 stddev on seq 05"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 05 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.025 --std_dev_noise 1.2 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 109 of 127 "
echo "runfile: doing run for 0.017 outliers and 4.5 stddev on seq 06"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 06 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.017 --std_dev_noise 4.5 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 110 of 127 "
echo "runfile: doing run for 0.034 outliers and 0.75 stddev on seq 02"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 02 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.034 --std_dev_noise 0.75 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 111 of 127 "
echo "runfile: doing run for 0.0 outliers and 6.0 stddev on seq 06"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 06 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.0 --std_dev_noise 6.0 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 112 of 127 "
echo "runfile: doing run for 0.025 outliers and 5.625 stddev on seq 02"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 02 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.025 --std_dev_noise 5.625 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 113 of 127 "
echo "runfile: doing run for 0.05 outliers and 2.625 stddev on seq 00"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 00 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.05 --std_dev_noise 2.625 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 114 of 127 "
echo "runfile: doing run for 0.059 outliers and 3.375 stddev on seq 06"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 06 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.059 --std_dev_noise 3.375 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 115 of 127 "
echo "runfile: doing run for 0.042 outliers and 3.375 stddev on seq 05"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 05 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.042 --std_dev_noise 3.375 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 116 of 127 "
echo "runfile: doing run for 0.042 outliers and 4.875 stddev on seq 05"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 05 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.042 --std_dev_noise 4.875 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 117 of 127 "
echo "runfile: doing run for 0.05 outliers and 0.0 stddev on seq 02"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 02 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.05 --std_dev_noise 0.0 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 118 of 127 "
echo "runfile: doing run for 0.034 outliers and 0.9 stddev on seq 05"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 05 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.034 --std_dev_noise 0.9 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 119 of 127 "
echo "runfile: doing run for 0.034 outliers and 2.25 stddev on seq 05"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 05 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.034 --std_dev_noise 2.25 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 120 of 127 "
echo "runfile: doing run for 0.017 outliers and 2.625 stddev on seq 02"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 02 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.017 --std_dev_noise 2.625 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 121 of 127 "
echo "runfile: doing run for 0.017 outliers and 4.875 stddev on seq 05"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 05 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.017 --std_dev_noise 4.875 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 122 of 127 "
echo "runfile: doing run for 0.0 outliers and 1.5 stddev on seq 00"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 00 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.0 --std_dev_noise 1.5 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 123 of 127 "
echo "runfile: doing run for 0.075 outliers and 1.05 stddev on seq 05"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 05 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.075 --std_dev_noise 1.05 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 124 of 127 "
echo "runfile: doing run for 0.059 outliers and 2.625 stddev on seq 02"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 02 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.059 --std_dev_noise 2.625 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 125 of 127 "
echo "runfile: doing run for 0.042 outliers and 1.2 stddev on seq 06"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 06 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.042 --std_dev_noise 1.2 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 126 of 127 "
echo "runfile: doing run for 0.059 outliers and 0.75 stddev on seq 02"
time python ./pointcloud_cluster_testing_measurement.py \
	--sequence 02 --skip_every_n 1 \
	--skip_vis true --skip_graphs true \
	--mode test_real_and_synthetic_landmark_association \
	--use_semantic no --clustering_override birch --cluster_target_num 120 \
	--gen_pcent_ouliers 0.059 --std_dev_noise 0.75 \
	--alignment_output ./results_output/collected_alignment_stats/detailed_outliers_kitti_3/ 
echo "runfile: done run 127 of 127 "
echo "end of runfile"