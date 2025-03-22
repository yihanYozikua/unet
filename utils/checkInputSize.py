from skimage import io

# Load the image
image = io.imread('../brain_tumor_seg/data/raw_data/GLI_train/BraTS-GLI-00070-000/BraTS_GLI_00000_000_70.png')

# Get the dimensions (height, width, channels)
height, width = image.shape[:2]
channels = image.shape[2] if image.ndim == 3 else 1

print(f"Height: {height}, Width: {width}, Channels: {channels}")