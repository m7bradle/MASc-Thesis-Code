you should put data from the kitti dataset here
the expected folder structure given the zips kitti provides (though we don't use all of the data) is:
kitti_dataset (this folder)
|-README.txt (this file)
\-dataset
	|-poses
	|	|-00.txt
	|	|-01.txt
	|	|-02.txt
	|	\-...
	|
	\-sequences
		|-00
		| |-calib.txt
		| |-times.txt
		| |-image_0
		| |		|-000000.png
		| |		|-000001.png
		| |		\-...
		| |-image_1
		| |		|-000000.png
		| |		|-000001.png
		| |		\-...
		| \-velodyne
		|		|-000000.bin
		|		|-000001.bin
		|		\-...
		|-01
		| |-calib.txt
		| |-times.txt
		| \-...
		|
		|-02
		| |-calib.txt
		| |-times.txt
		| \-...
		|
		\-...
		
We predominantly rely on data from sequences 00, 02, 05, 06, and occasionally 07
