
import cv2
import numpy as np
from math import floor, isnan
from statistics import mean
import json
import os
from scipy.spatial import KDTree
import matplotlib.pyplot as plt



class sample_collection:
    def __init__(self):
        self.samples = []

    def add(self, sample):
        self.samples.append(sample)

    def to_dict(self):
        return [samp.__dict__ for samp in self.samples]

class sample:
    def __init__(self, origin=None):
        self.origin = origin
        self.score = 0
        self.score_normalized = 0
        self.points = []
        self.kd_tree = None

    def add_point(self, point):
        self.points.append(point)

    def to_dict(self):
        return {
            'origin': self.origin.to_dict() if self.origin else None,
            'score': self.score,
            'points': [point.to_dict() for point in self.points]
        }

class point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.dist_to_nearest_neighbor = None
        self.inlier = False

    def to_tuple(self):
        return (self.x, self.y)

    def to_dict(self):
        return {
            'x': self.x,
            'y': self.y,
            'dist_to_nearest_neighbor': self.dist_to_nearest_neighbor
        }
    
    def __str__(self):
        return f"{self.x},{self.y}"



def sample_visible_objects(image, points, detection_colors):
    
    # Get the dimensions of the image
    height, width = image.shape
    
    # List to store the nearest non-white pixels
    nearest_pixels = sample_collection()
    
    for p in points:
        samp = sample(origin=p)
        # Iterate through 360 degrees
        for angle in range(0, 360, 5):
            radian = np.deg2rad(angle)
            
            # Iterate outward from the center point
            for radius in range(max(height, width)):
                x = int(p.x + radius * np.cos(radian))
                y = int(p.y + radius * np.sin(radian))
                
                # Check if the point is within the image bounds
                if x >= 0 and x < width and y >= 0 and y < height:
                    # Check if the pixel is non-white
                    #if gray_image[y, x] < 255:
                    if image[y, x] in detection_colors:
                            x += round(np.random.normal(0,1))
                            y += round(np.random.normal(0,0.5))
                            samp.add_point(point(x=x, y=y))
                            break
        nearest_pixels.add(samp)
            
    
    return nearest_pixels

    

def get_pixel_colors(image):

    # Get the dimensions of the image
    height, width, _ = image.shape

    # Create a list to store pixel colors
    pixel_colors = []


    # Iterate through each pixel
    for y in range(height):
        for x in range(width):
            # Get the color of the pixel (BGR format)
            color = image[y, x]
            pixel_colors.append(color)

    return pixel_colors

def get_free_points(image, fraction):

    # Get the dimensions of the image
    height, width = image.shape

    points = []

    # Iterate through each pixel
    for y in range(0, height, floor(height/fraction)):
        for x in range(0, width, floor(width/fraction)):
            # Get the color of the pixel (BGR format)
            if image[y, x] == 255:
                points.append(point(x,y))

    return points


def display_result(image, display_size, title="image"):
    # Display the image

    resized_image = cv2.resize(image, (display_size, display_size))
    cv2.imshow(title, resized_image)


def min_max_normalize(data):
    # Calculate the minimum and maximum values in the list
    min_val = min(data)
    max_val = max(data)
    
    # Perform min-max normalization
    normalized_data = [(x - min_val) / (max_val - min_val) for x in data]
    
    return normalized_data


def load_image():
    image = cv2.imread('edi/test_maps/map_1.png', cv2.IMREAD_GRAYSCALE)
    if image is None:
        print('Could not open or find the image: ')
        exit(0)
    return image

def load_config():
    # Open and read the JSON file
    with open('edi/config.json', 'r') as file:
        data = json.load(file)
    return data

def get_display_image(image, config):

    if not is_greyscale(image=image):
        print("Error: input to get_display_image() must be greyscale")
        exit()

    result = convert_to_bgr(image)
    height, width = image.shape

    for y in range(height):
        for x in range(width):
            g_val = image[y, x]
            color_val = get_color_info(config=config, greyscale_value=g_val)
            if color_val == None:
                result[y, x] = [0, 160, 255] # Bright orange to show error (None)
            else:
                result[y, x] = color_val['BGR_value']

    return result


def is_greyscale(image):
    return len(image.shape)<3

def convert_to_bgr(image):
    if is_greyscale(image):
        return cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    else:
        print("Warning: Image already BGR")
        return image

def get_color_info(config, greyscale_value):
    return config["color_map"].get(str(greyscale_value), None)


def load_json_file(file_path):
    # Open and read the JSON file
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    # Convert the JSON data to a Python object
    sample_collection_obj = sample_collection()
    for item in data:
        samp = sample(origin=point(item['origin']['x'], item['origin']['y']))
        samp.score = item['score']
        for p in item['points']:
            samp.add_point(point(x=p['x'], y=p['y']))
        sample_collection_obj.add(samp)
    return sample_collection_obj



def load_data(laser_data_path, map_data_path, map):
    if os.path.exists(laser_data_path):
        laser_sample_collection = load_json_file(laser_data_path)
    else:
        laser_sample_collection = generate_laser_data(map)

    if os.path.exists(map_data_path):
        map_sample_collection = load_json_file(map_data_path)
    else:
        map_sample_collection = generate_map_data(map)

    return laser_sample_collection, map_sample_collection




def save_data(data, filename):
    # Open and write the JSON file
    with open(filename, 'w') as file:
        json.dump(data.to_dict(), file, indent=4, default=lambda o: o.to_dict() if hasattr(o, 'to_dict') else o)


def generate_laser_data(map):
    sample_positions = get_free_points(image=map, fraction=40)
    laser_sample = sample_visible_objects(map, sample_positions, detection_colors=[0, 2])

    return laser_sample

def generate_map_data(map):
    sample_positions = get_free_points(image=map, fraction=40)
    map_sample = sample_visible_objects(map, sample_positions, detection_colors=[0, 3])

    return map_sample


def color_point_radius(image, point, color, radius=1):
    cv2.circle(image, (point.x, point.y), radius, color, -1)
    return image

def create_point_list_from_sample_collection(sample_collection):
    # Extract all points from a sample_collection
    all_points = []
    for sample in sample_collection.samples:
        for p in sample.points:
            all_points.append((p.x, p.y))
    
    return all_points


def calculate_sample_score_inlier_percentage(sample, kd_tree, conf):

    distances = []
    inliers = 0
    outliers = 0
    for point in sample.points:
        distance, _ = kd_tree.query(point.to_tuple())
        distances.append(distance)
        point.dist_to_nearest_neighbor = distance
        if distance > conf["parameters"]["noise_threshold_distance"]:
            point.inlier = True
            inliers += 1
        else:
            outliers += 1
    
    if outliers == 0 and inliers == 0:
        return None
    if outliers == 0 and inliers > 0:
        return 1
   
    return (inliers/outliers)
        


def calculate_sample_score_rms(sample, kd_tree):
    """
    Calculate the root mean square (RMS) distance from each point in the sample
    to its nearest neighbor in the KDTree and set the sample's score.

    Args:
        sample (sample): The sample object containing points.
        kd_tree (KDTree): The KDTree built from map_sample_collection points.
    """
    distances = []
    for point in sample.points:
        distance, _ = kd_tree.query(point.to_tuple())
        distances.append(distance)
        point.dist_to_nearest_neighbor = distance
    
    # Calculate the RMS distance
    if distances:
        rms_distance = np.sqrt(np.mean(np.square(distances)))
    else:
        rms_distance = None  # Default to None if no points are present

    return rms_distance


def add_gaussian_circle(image, laser_sample_collection, radius):

    result = image.copy()
    # result = cv2.GaussianBlur(result,(5,5),0)

    for sample in laser_sample_collection.samples:
        # Draw a Gaussian circle around the sample's origin
        for y in range(-radius, radius + 1):
            for x in range(-radius, radius + 1):
                distance = np.sqrt(x**2 + y**2)
                if distance <= radius:
                    intensity = np.exp(-distance**2 / (2 * (radius / 2)**2))  # Gaussian falloff
                    px = sample.origin.x + x
                    py = sample.origin.y + y
                    if 0 <= px < image.shape[1] and 0 <= py < image.shape[0]:
                        if image[py,px] > 3:
                            result[py, px] = int(result[py, px] * (1 - intensity) + 255*(1-intensity))



    return result


def add_gaussian_circle_old(image, laser_sample_collection, radius):

    result = image.copy()
    result = convert_to_bgr(result)

    for sample in laser_sample_collection.samples:

        # Convert normalized score to a heatmap color (purple-blue to red)
        color = (
            int(255 * (1 - sample.score_normalized)),  # Blue channel
            int(0),                             # Green channel
            int(255 * sample.score_normalized)         # Red channel
        )

        # Draw a Gaussian circle around the sample's origin
        for y in range(-radius, radius + 1):
            for x in range(-radius, radius + 1):
                distance = np.sqrt(x**2 + y**2)
                if distance <= radius:
                    intensity = np.exp(-distance**2 / (2 * (radius / 2)**2))  # Gaussian falloff
                    px = sample.origin.x + x
                    py = sample.origin.y + y
                    if 0 <= px < image.shape[1] and 0 <= py < image.shape[0]:
                        # Blend the Gaussian intensity with the existing pixel color
                        result[py, px] = (
                            np.clip(int(result[py, px][0] * (1 - intensity) + color[0] * intensity), 0, 255),
                            np.clip(int(result[py, px][1] * (1 - intensity) + color[1] * intensity), 0, 255),
                            np.clip(int(result[py, px][2] * (1 - intensity) + color[2] * intensity), 0, 255)
                        )
    return result


def calculate_score(sample, kd_tree):
    scores = []
    for sample in sample_collection.samples:

        if sample.origin.x == 64 and sample.origin.y == 192:
            print("missing obj:")

        sample.score = calculate_sample_score(sample, kd_tree) # Add score to sample
        scores.append(sample.score) # Collect scores in a list
        print(f"sample: {sample.origin}, points used: {len(sample.points)}, score: {sample.score}")

    return scores

def main():

    conf = load_config()
    map = load_image()
    laser_sample_collection, map_sample_collection = load_data("edi/data/laser_sample.json", "edi/data/map_sample.json", map)

    if (len(laser_sample_collection.samples) != len(map_sample_collection.samples)):
        print("Error: Unequal number of sample points!")
        exit()

    no_samples = len(laser_sample_collection.samples)
    scores = []

    # Get map points for sample
    for i in range(no_samples):
        map_sample = map_sample_collection.samples[i]
        map_sample_points = []
        for p in map_sample.points:
            map_sample_points.append((p.x, p.y))

        # Build tree for given sample
        kd_tree = KDTree(map_sample_points)

        # Calculate sample score
        score = calculate_sample_score_inlier_percentage(laser_sample_collection.samples[i], kd_tree, conf)
        scores.append(score)


    distances = []
    for sample in laser_sample_collection.samples:
        for p in sample.points:
            distances.append(p.dist_to_nearest_neighbor)

    # print(len(distances))
    # plt.hist(distances, bins=50, color='skyblue', edgecolor='black')
    # plt.xlabel('Values')
    # plt.ylabel('Frequency')
    # plt.title('Basic Histogram')
    # plt.show()
    # Normalize the scores

    normalized_scores = min_max_normalize(scores)
    for i in range(len(scores)):
        laser_sample_collection.samples[i].score_normalized = normalized_scores [i]


    heat_map = map.copy()
    heat_map = add_gaussian_circle_old(heat_map, laser_sample_collection, radius=8)




    for s in laser_sample_collection.samples:
        color_point_radius(image=map, point=s.origin, color=6, radius=1)


    for s in map_sample_collection.samples:
        for p in s.points:
            color_point_radius(map, p, 5)

    for s in laser_sample_collection.samples:
        for p in s.points:
            color_point_radius(map, p, 4)


    save_data(laser_sample_collection, "edi/data/laser_sample.json")
    save_data(map_sample_collection, "edi/data/map_sample.json")

    display_image = get_display_image(map, conf)
    display_result(display_image, conf["parameters"]["display_window_size"], title="disp_img")
    display_result(heat_map, conf["parameters"]["display_window_size"], title="heat_map")



    # Wait for a key press and close the image window
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Using the special variable 
# __name__
if __name__=="__main__":
    main()