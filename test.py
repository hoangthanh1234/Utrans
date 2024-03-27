import numpy as np 

#sem_label = np.fromfile('./dataset/nuScenes/lidarseg/v1.0-mini/0ab9ec2730894df2b48df70d0d2e84a9_lidarseg.bin', dtype=np.uint8).reshape((-1, 1))
#print(sem_label)

#raw_data = np.fromfile('./dataset/nuScenes/samples/LIDAR_TOP/n008-2018-05-21-11-06-59-0400__LIDAR_TOP__1526915243047392.pcd.bin', dtype=np.float32).reshape((-1, 5))
#print("pointxyz: ", raw_data[:,:3])
#print("intensity_values : ", raw_data[:,3])
#sem_label_raw = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
#pointcloud = raw_data[:, :4]

image_height = 64  # Assuming an image height of 32 pixels
image_width = 384   # Assuming an image width of 32 pixels

patch_stride = [2,8]



output_height = int(np.ceil(image_height / patch_stride[0]))
output_width = int(np.ceil(image_width / patch_stride[1]))

# Print the output shape
print(f"Output shape after patch embedding: {(output_height, output_width)}")
