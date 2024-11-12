import matplotlib.pyplot as plt
import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

def vis_area(x_points = [
    4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 
    144, 169, 196, 225, 256, 289, 400, 625],
y_points = [
    43.2, 54.7, 62.0, 66.6, 66.7, 68.6, 69.6, 
    71.3, 74.1, 71.4, 73.2, 72.9, 72.9, 73.1, 
    72.1, 71.8, 73.6, 72.4
], file_path="finetuning_cropping_factor.png"):
	plt.figure(figsize=(10, 6))
	plt.plot(x_points, y_points, marker='o', linestyle='-', color='b')
	plt.xlabel("Prev bbox's area is enalarged in N times")
	plt.ylabel("IoU VOT22ST (63 seqs)")
	plt.title("Tracking by cropping")
	
	# plt.ylim(40, 75)
	
	y_ticks = plt.yticks()[0].tolist()  # Get existing y-tick values
	y_ticks.append(73)  # Add 73 to y-ticks
	plt.yticks(sorted(y_ticks))

	plt.axhline(y=73, color='r', linestyle='--', label="baseline")

	plt.legend()

	plt.grid(True)
	plt.savefig(file_path)

def vis_regressed_parabola(file_path=None):
	# Define the three points
	points = np.array([[9, 71.3], [10, 74.1], [11, 71.4]])  # replace these with your specific points
	x_points = points[:, 0]
	y_points = points[:, 1]

	# Set up the system of equations for a parabola y = ax^2 + bx + c
	A = np.vstack([x_points**2, x_points, np.ones(len(x_points))]).T
	a, b, c = np.linalg.solve(A, y_points)

	# Generate x values for the parabola plot
	x_fit = np.linspace(min(x_points) - 1, max(x_points) + 1, 100)
	y_fit = a * x_fit**2 + b * x_fit + c

	vertex_x = -b*1.0/(2*a)
	vertex_y = a * (vertex_x**2) + b * vertex_x + c

	# Plot the points and the parabola
	plt.scatter(x_points, y_points, color='red', label=f'estimation for three points: vertex ({vertex_x:.2f}, {vertex_y:.2f})')
	plt.plot(x_fit, y_fit, color='blue', label=f'parabola: $y = {a:.2f}x^2 + {b:.2f}x + {c:.2f}$')
	plt.ylabel('IoU')
	plt.xlabel("prev bbox's area is enalarged in N times")
	plt.legend()
	plt.title('parabola regression')
	plt.grid(True)

	# Save the plot as an image
	plt.savefig(file_path)

	# Show the plot
	plt.show()


def vis_oracle(file_path=None):
	# Define the three points
	# points = np.array([[5, 63], [10, 57], [20, 39], [30, 24], [40, 10], [50, 1]]) # marathon
	# points = np.array([[5, 40.28], [10, 35.42], [20, 31.94], [30, 25.69], [40, 18.75], [50, 6.94]]) # hand2
	# points = np.array([[5, 29.95], [10, 26.37], [20, 22.80], [30, 18.96], [40, 12.36], [50, 4.12]]) # shaking
	# points = np.array([[5, 32], [10, 21], [20, 0], [30, 0], [40, 0], [50, 0]]) # marathon RR+oracle
	# points = np.array([[5, 19.44], [10, 13.19], [20, 4.86], [30, 3.47], [40, 1.39], [50, 0]]) # hand2 RR+oracle
	# points = np.array([[5, 30.77], [10, 29.67], [20, 26.37], [30, 21.70], [40, 10.44], [50, 2.20]]) # shaking RR+oracle
	# points = np.array([[5, 36.17], [10, 14.63], [20, 3.72], [30, 0.53], [40, 0], [50, 0]]) # handball1 
	# points = np.array([[5, 37.5], [10, 15.96], [20, 3.19], [30, 1.06], [40, 0], [50, 0]]) # handball1 RR+oracle
	# points = np.array([[5, 0], [10, 0], [20, 0], [30, 0], [40, 0], [50, 0]]) # singer3 RR+oracle
	# points = np.array([[5, 45.99], [10, 32.85], [20, 24.09], [30, 11.68], [40, 5.84], [50, 0.73]]) # soldier orcale
	points = np.array([[5, 36.50], [10, 25.55], [20, 18.98], [30, 13.14], [40, 6.57], [50, 1.46]]) 


	x_points = points[:, 0]
	y_points = points[:, 1]


	# Plot the points and the parabola
	plt.scatter(x_points, y_points, color='red', label='')
	plt.plot(x_points, y_points, color='blue', label='')
	plt.ylabel("error (%): precicted mask is not the best")
	plt.xlabel("thr (%): argmax_i(IoU(GT, mask_i)) - IoU(GT, mask_pred_sam2) > thr")

	plt.yticks(y_points)
	plt.xticks(x_points)
	plt.title('oracle+RR experiment for soldier')
	plt.grid(True)

	# Save the plot as an image
	plt.savefig(file_path)

	# Show the plot
	plt.show()


def create_video():
	# Set the path to the directory containing the frames
	frames_path = '/datagrid/personal/rozumrus/BP_dg/vot22ST/sequences/hand2/color'  # Replace with the path to your frames
	output_video_path = 'hand2_output.mp4'
	frame_rate = 24  # Set the desired frame rate (e.g., 24 fps)

	# Get all image file names in the frames directory
	frame_files = sorted(
	    [f for f in os.listdir(frames_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
	)

	frame_files = [i for i in frame_files if i[0] != '.']

	# Ensure there are frames to create a video
	if not frame_files:
	    raise ValueError("No frames found in the specified directory.")

	# Get the size of the frames
	first_frame = cv2.imread(os.path.join(frames_path, frame_files[0]))
	height, width, _ = first_frame.shape

	# Define the video codec and create VideoWriter object
	fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4
	out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

	# Write each frame to the video
	for frame_file in frame_files:
	    frame = cv2.imread(os.path.join(frames_path, frame_file))
	    out.write(frame)

	# Release the video writer object
	out.release()

	print(f"Video saved as {output_video_path}")


parser = argparse.ArgumentParser()
parser.add_argument('--vis_area', action="store_true")
parser.add_argument('--vis_oracle', action="store_true")
parser.add_argument('--vis_parabola', action="store_true")
parser.add_argument('--path',default="output.png")
args = parser.parse_args()

if args.vis_area:
	vis_area(file_path=args.path)
elif args.vis_parabola:
	vis_regressed_parabola(file_path=args.path)
elif args.vis_oracle:
	vis_oracle(file_path=args.path)


# create_video()