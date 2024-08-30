#!/bin/bash
wget -r -H -nc -np -nH --cut-dirs=1 -e robots=off -l1 -i ./list_of_zips.txt -B http://archive.org/download/semantic_mask_kitti_cityscapes/

unzip ./semantic_mask_kitti_cityscapes/seq00_semantic_mask_output_cityscape_key.zip -d .
unzip ./semantic_mask_kitti_cityscapes/seq02_semantic_mask_output_cityscape_key.zip -d .
unzip ./semantic_mask_kitti_cityscapes/seq05_semantic_mask_output_cityscape_key.zip -d .
unzip ./semantic_mask_kitti_cityscapes/seq06_semantic_mask_output_cityscape_key.zip -d .
unzip ./semantic_mask_kitti_cityscapes/seq07_semantic_mask_output_cityscape_key.zip -d .

