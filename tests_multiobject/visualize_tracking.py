import json
import matplotlib.pyplot as plt
import sys

def visualize_tracking_points(json_file, output_image_file):
    # Load the JSON data from the file
    with open(json_file, 'r') as file:
        data = json.load(file)
    
    # Extract the relevant points from the JSON structure
    # Assuming the points are in the second analysis result
    points = data['results']['baseline']['results'][1][0][0]
    
    # Separate the points into two lists: thresholds and values
    thresholds = [point[0] for point in points]
    values = [point[1] for point in points]
    
    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, values, marker='o', linestyle='-', color='b')
    
    # Labeling the axes and the plot
    plt.xlabel('Threshold')
    plt.ylabel('Accuracy/Success Rate')
    plt.title('Tracker Performance Over Time/Thresholds')
    plt.grid(True)
    
    # Save the plot to a file
    plt.savefig(output_image_file)
    
    # Optionally, display the plot
    # plt.show()

if __name__ == "__main__":
    # Check if the correct number of arguments is passed
    if len(sys.argv) != 3:
        print("Usage: python script_name.py <input_json_file> <output_image_file>")
    else:
        json_file = sys.argv[1]
        output_image_file = sys.argv[2]
        visualize_tracking_points(json_file, output_image_file)

