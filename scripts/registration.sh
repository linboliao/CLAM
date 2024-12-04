export PYTHONPATH=/data2/yhhu/LLB/Code/CLAM:$PYTHONPATH
#python ../data_process/image_registration.py --data_root /data2/yhhu/LLB/Data/MSI/ --patch_size 1024 --ihc_ext PMS2
python ../data_process/image_registration.py --data_root /data2/yhhu/LLB/Data/前列腺癌数据/CKPan/ --patch_size 4096 --ihc_ext CK --check True