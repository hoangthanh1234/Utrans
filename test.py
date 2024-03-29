import numpy as np 

#sem_label = np.fromfile('./dataset/nuScenes/lidarseg/v1.0-mini/0ab9ec2730894df2b48df70d0d2e84a9_lidarseg.bin', dtype=np.uint8).reshape((-1, 1))
#print(sem_label)

#raw_data = np.fromfile('./dataset/nuScenes/samples/LIDAR_TOP/n008-2018-05-21-11-06-59-0400__LIDAR_TOP__1526915243047392.pcd.bin', dtype=np.float32).reshape((-1, 5))
#print("pointxyz: ", raw_data[:,:3])
#print("intensity_values : ", raw_data[:,3])
#sem_label_raw = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
#pointcloud = raw_data[:, :4]

image_height = 16  # Assuming an image height of 32 pixels
image_width = 32   # Assuming an image width of 32 pixels

patch_stride = [2,8]



output_height = int(np.ceil(image_height / patch_stride[0]))
output_width = int(np.ceil(image_width / patch_stride[1]))

# Print the output shape
print(f"Output shape after patch embedding: {(output_height, output_width)}")


# (decoder): DecoderUpConv(
#       (up_conv_block): UpConvBlock(
#         (conv_upsample): Sequential(
#           (0): Conv2d(384, 4096, kernel_size=(1, 1), stride=(1, 1))
#           (1): Rearrange('b (c s0 s1) h w -> b c (h s0) (w s1)', s0=2, s1=8)
#         )
#         (conv1): Sequential(
#           (0): Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (1): LeakyReLU(negative_slope=0.01)
#           (2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         )
#         (conv_output): Sequential(
#           (0): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
#           (1): LeakyReLU(negative_slope=0.01)
#           (2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         )
#       )
#     )
#     (kpclassifier): KPClassifier(
#       (kpconv): KPConv(radius: 0.60, in_feat: 256, out_feat: 256)
#       (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU()
#       (head): Conv2d(256, 17, kernel_size=(1, 1), stride=(1, 1))
#     )
