from natsort import natsorted
import imageio.v2 as imageio
import os

# Directory containing the images
image_dir = '/home/taoletian/epsilon-controllability/figs/MassSpringwoControl/[-0.  0.]state-0.05epsilon-10000samples-20231221-123100/epsilon_controllable_set'

# Output video file
video_file = '/home/taoletian/epsilon-controllability/figs/MassSpringwoControl/[-0.  0.]state-0.05epsilon-10000samples-20231221-123100/epsilon_controllable_set/video.mp4'

# Get the file names of the PNG images
images = [os.path.join(image_dir, file) for file in os.listdir(image_dir) if file.endswith('.png')]
images = natsorted(images)

# Create a writer object
writer = imageio.get_writer(video_file, fps=5)

# Add images to the video
for image in images:
    writer.append_data(imageio.imread(image))

# Close the writer
writer.close()
