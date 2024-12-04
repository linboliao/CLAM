export PYTHONPATH=/data2/yhhu/LLB/Code/CLAM:$PYTHONPATH
#python ../data_process/patch_coord_generate.py --data_root /data2/yhhu/LLB/Data/前列腺癌数据/CKPan/ --patch_size 4096 --overlap 0.4 --patch_level 1
#python ../data_process/patch_coord_generate.py --data_root /data2/yhhu/LLB/Data/前列腺癌数据/数据验证/ --patch_size 256 --overlap 0
python ../data_process/patch_coord_generate.py --data_root /data2/yhhu/LLB/Data/前列腺癌数据/test/ --patch_size 256 --overlap 0