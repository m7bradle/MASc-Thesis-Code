#!/bin/bash
set -e

# for sequence 02
time python ./pointcloud_cluster_testing_measurement.py --mode compare_match_pairs_clusters_alignment_measurement --sequence 02 \
--skip_every_n 1 --use_semantic no --clustering_override dbscan --cluster_target_num 120 \
--skip_vis true --skip_graphs true \
--cluster_output ./results_output/collected_cluster_stats/total_correction/plain_dbscan/sequence_02/

time python ./pointcloud_cluster_testing_measurement.py --mode compare_match_pairs_clusters_alignment_measurement --sequence 02 \
--skip_every_n 1 --use_semantic no --clustering_override birch --cluster_target_num 120 \
--skip_vis true --skip_graphs true \
--cluster_output ./results_output/collected_cluster_stats/total_correction/plain_birch/sequence_02/

time python ./pointcloud_cluster_testing_measurement.py --mode compare_match_pairs_clusters_alignment_measurement --sequence 02 \
--skip_every_n 1 --use_semantic yes --clustering_override dbscan --cluster_target_num 20 \
--skip_vis true --skip_graphs true \
--cluster_output ./results_output/collected_cluster_stats/total_correction/semantic_dbscan/sequence_02/

time python ./pointcloud_cluster_testing_measurement.py --mode compare_match_pairs_clusters_alignment_measurement --sequence 02 \
--skip_every_n 1 --use_semantic yes --clustering_override birch --cluster_target_num 20 \
--skip_vis true --skip_graphs true \
--cluster_output ./results_output/collected_cluster_stats/total_correction/semantic_birch/sequence_02/
