为了减少各种文件地址参数，坐标提取、patch 提取、特征提取等等使用统一的文件结构

dataroot: 该目录下的文件，提取坐标默认按照下面结果存储

	--slides : wsi 文件存储位置

	--patch/{patch_size}：patch数据 按patch size分类

		--coord：提取坐标

		--mask:

		--stitch: 

		--count.csv

	--features/{patch_size}/{model_name}: 

		--h5_files:  h5py格式

		--pt_files: pt格式

	--labels：标签格式

		--label.csv

	--points: 配准使用数据

	--results

		--heatmaps：热力图结果

		--test：测试集合结果

		--train：训练结果，多折训练

