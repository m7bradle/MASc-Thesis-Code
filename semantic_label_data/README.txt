The folders in this directory contain the computed semantic masks for frames in each sequence of kitti.
You'll need to run download.sh to download them from the internet archive and unpack them.
The result is a series of numpy files that contain pixel-wise semantic labels for each frame.
These *.npy files don't require pickle to load (using np.load()) and so should be safe. At any rate, pickle has been disabled by default in np.load() for some time now due to past CVEs.
see https://numpy.org/doc/stable/reference/generated/numpy.load.html
