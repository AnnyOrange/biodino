import tifffile

# 读取图片
img_path = "/mnt/deepcad_nfs/000-LM-dataset-preprocessed/idr0116-deboer-npod/20210805-Globus/norm_6158-265b.ome.tif"
with tifffile.TiffFile(img_path) as tif:
    # 查看TIFF属性
    print(f"Series count: {len(tif.series)}")
    
    # 查看第一个系列的shape
    series = tif.series[0]
    print(f"Image shape: {series.shape}")  # (pages, channels, height, width) 或类似
    print(f"Data type: {series.dtype}")
    
    # 读取数据
    img_data = series.asarray()
    print(f"Full array shape: {img_data.shape}")
    
    # 如果是OME-TIFF，可以查看元数据
    if tif.ome_metadata is not None:
        print("OME metadata available")