import numpy as np
from scipy.ndimage import convolve, gaussian_filter


def generate_kernels(voxel_size):
    k_z = np.zeros((3, 1, 1), dtype=np.float32)
    k_z[0, 0, 0] = voxel_size[2]
    k_z[2, 0, 0] = -voxel_size[2]

    k_y = np.zeros((1, 3, 1), dtype=np.float32)
    k_y[0, 0, 0] = voxel_size[1]
    k_y[0, 2, 0] = -voxel_size[1]

    k_x = np.zeros((1, 1, 3), dtype=np.float32)
    k_x[0, 0, 0] = voxel_size[0]
    k_x[0, 0, 2] = -voxel_size[0]

    return k_z, k_y, k_x


def get_descriptor(mask, sigma, voxel_size):

    sigma_scaled = sigma/np.array(voxel_size, dtype=np.float32)
    soft_mask = gaussian_filter(mask, sigma_scaled, mode='constant', output=np.float32)
    
    k_z, k_y, k_x = generate_kernels(voxel_size)

    d_z = convolve(soft_mask, k_z, mode='constant')
    d_y = convolve(soft_mask, k_y, mode='constant')
    d_x = convolve(soft_mask, k_x, mode='constant')

    d_zz = convolve(d_z, k_z, mode='constant')
    d_yy = convolve(d_y, k_y, mode='constant')
    d_xx = convolve(d_x, k_x, mode='constant')
    d_zy = convolve(d_z, k_y, mode='constant')
    d_zx = convolve(d_z, k_x, mode='constant')
    d_yx = convolve(d_y, k_x, mode='constant')

    d = np.stack([d_z, d_y, d_x,
                  d_zz, d_yy, d_xx,
                  d_zy, d_zx, d_yx])

    # normalize
    soft_mask[soft_mask==0] = 1
    d = d/soft_mask + 0.5

    lsds = np.concatenate([d, soft_mask[None, :]])
    lsds[:, soft_mask==0] = 0

    return lsds
