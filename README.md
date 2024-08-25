# MASc-Thesis-Code
Code relevant to reproducing the results of my MASc thesis.

## Setup of dependencies
To install the needed packages, run `source install.source` which will set up a Conda env and install all the packages needed using pip inside it. Running `source activate.source` will activate the conda env.

These could probably be installed in a similar way in a regular pip env if you prefer. Conda was simply easier during development for some CUDA-related packages, and has support for nested pip usage now.

You will need to obtain a copy of GrassGraph. We provide a lightly modified version on github with some needed configuration changes to quash graphs from popping up during runtime. You can obtain it by running `source clone_grassgraphs.source`.

## Downloading the Kitti dataset and semantic labels
You will need to obtain the Kitti dataset from https://www.cvlibs.net/datasets/kitti/, specifically sequences 00, 02, 05, and 06 (you may also want 07). See `./kitti_dataset/README.txt` for details.

You will also need to obtain the semantic labels we generated for the kitti dataset. They can be obtained from the Internet Archive by running `./semantic_label_data/download.sh`. See `./semantic_label_data/README.txt` for details. 

We provide captured results from running DSO on KITTI for convenience, so it is not required to compile and run the modified version of DSO required. See `./kitti_dso_files/README.txt` for details.

## Launching the first set of experiments
The first set of results which characterized generated landmarks themselves can be launched once setup is complete using various `characterizing_landmarks_sequence_*_compensated.sh` and `characterizing_landmarks_sequence_*_uncompensated.sh` scripts.

## Launching the Second set of experiments
The second set of results which investigated the feasibility of using GrassGraph to associate these landmarks are meant to be run in parallel, for brevity. By default 8 threads are specified, with runfiles to be run being `measure_assoc_perf_kitti_*.sh` and `measure_detailed_outliers_kitti_*.sh` and `measure_assoc_perf_shrec_*.sh`. The python scripts `gen_sensitivity_runfiles_noise_and_outliers.py` and `gen_sensitivity_runfiles_detailed_outliers.py` and `gen_sensitivity_runfiles_noise_and_outliers_shrec_data.py` can be used to generate new runfiles for a different number of concurrent jobs.

## Generating results and where to find them
The results from the first set of experiments are generated as part of running them and require no further preparation. In particular, they can be found in `./results_output/` at:
```
./results_output/collected_cluster_stats/no_total_correction/*/sequence_*/nn_distance_between_histogram.png
./results_output/collected_cluster_stats/no_total_correction/*/sequence_*/nn_precent_outliers_histogram.png
./results_output/collected_cluster_stats/total_correction/*/sequence_*/nn_distance_between_histogram.png
./results_output/collected_cluster_stats/total_correction/*/sequence_*/nn_precent_outliers_histogram.png
```
for graphs, and:
```
./results_output/collected_cluster_stats/no_total_correction/*/sequence_*/logfile.txt
./results_output/collected_cluster_stats/total_correction/*/sequence_*/logfile.txt
```
for numeric results.

The results from the second set require some further collection and processing after all the concurrent jobs have been run to completion. Run `generate_alignment_results.sh` to collect the results from all of the threads, and then to generate the final graphs and tabular results run `python collect_assoc_measures.py basic_analysis_kitti` and `python collect_assoc_measures.py detailed_outliers_kitti` and `python collect_assoc_measures.py basic_analysis_shrec`.

The results files themselves can then be found in `./results_output/`, particularly:
```
./results_output/allignment_stats_analysis_results/basic_analysis_kitti/pcent_match/pcent_match_avg_lineplot.png
./results_output/allignment_stats_analysis_results/basic_analysis_kitti/matrix_frob/matrix_frob_med_lineplot.png
./results_output/allignment_stats_analysis_results/basic_analysis_kitti/angle_difference/angle_diff_avg_lineplot.png
./results_output/allignment_stats_analysis_results/detailed_outliers_kitti/pcent_match/pcent_match_avg_lineplot.png
./results_output/allignment_stats_analysis_results/detailed_outliers_kitti/matrix_frob/matrix_frob_med_lineplot.png
./results_output/allignment_stats_analysis_results/detailed_outliers_kitti/angle_difference/angle_diff_avg_lineplot.png
./results_output/allignment_stats_analysis_results/basic_analysis_shrec/pcent_match/pcent_match_avg_lineplot.png
./results_output/allignment_stats_analysis_results/basic_analysis_shrec/matrix_frob/matrix_frob_med_lineplot.png
./results_output/allignment_stats_analysis_results/basic_analysis_shrec/angle_difference/angle_diff_avg_lineplot.png
```
for graphs, and:
```
./results_output/allignment_stats_analysis_results/basic_analysis_kitti/pcent_match/pcent_match_avg_grid_tables.txt
./results_output/allignment_stats_analysis_results/basic_analysis_kitti/matrix_frob/matrix_frob_med_grid_tables.txt
./results_output/allignment_stats_analysis_results/basic_analysis_kitti/angle_difference/angle_diff_avg_grid_tables.txt
./results_output/allignment_stats_analysis_results/detailed_outliers_kitti/pcent_match/pcent_match_avg_grid_tables.txt
./results_output/allignment_stats_analysis_results/detailed_outliers_kitti/matrix_frob/matrix_frob_med_grid_tables.txt
./results_output/allignment_stats_analysis_results/detailed_outliers_kitti/angle_difference/angle_diff_avg_grid_tables.txt
./results_output/allignment_stats_analysis_results/basic_analysis_shrec/pcent_match/pcent_match_avg_grid_tables.txt
./results_output/allignment_stats_analysis_results/basic_analysis_shrec/matrix_frob/matrix_frob_med_grid_tables.txt
./results_output/allignment_stats_analysis_results/basic_analysis_shrec/angle_difference/angle_diff_avg_grid_tables.txt
```
for full numeric results as seen in the appendix.

The folder `./results_output/collected_alignment_stats/` will appear after running the second set of experiments but before generating the final results as above and serves as a temporary holding place for the results of each thread.

## Brief overview of some folders and files not mentioned above
The `numpy_saves` folder is used to cache some intermediary results, particularly the results of various data preparation steps.
The `matlab_scripts` folder contains some helper scripts and glue code for interacting with GrassGraph and fetching datasets (SHREC) bundled with it.

## License considerations
This repo is currently GPLv3 to ease compatibility with use of GrassGraph, though earlier versions which include useful dataset preparation code which predates use of GrassGraph and is not derived froom it are available under MIT terms.
