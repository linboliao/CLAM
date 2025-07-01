from wsi import WSIOperator

path = '/NAS2/Data4/llb/中日友好医院结直肠癌数据/LS/352908/352908_4_HE.sdpc'
wsi = WSIOperator(path)
width, height = wsi.level_dimensions[0]
print(width, height)