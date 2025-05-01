
import cv2
import numpy as np
from math import floor
from statistics import mean
import json
import os
from scipy.spatial import KDTree

class sample_collection_pair:
    def __init__(self, laser_sample_collection, map_sample_collection):
        self.laser_sample_collection = laser_sample_collection
        self.map_sample_collection = map_sample_collection

    def add(self, laser_sample, map_sample):
        self.laser_sample.add(laser_sample)
        self.map_sample.add(map_sample)

    def to_dict(self):
        return {
            'laser_sample': self.laser_sample.to_dict(),
            'map_sample': self.map_sample.to_dict()
        }


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
        self.points = []

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

    def to_tuple(self):
        return (self.x, self.y)

    def to_dict(self):
        return {
            'x': self.x,
            'y': self.y
        }

def euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def nearest_neighbor_distances(list1, list2):
    distances = []
    for point1 in list1:
        min_distance = float('inf')
        for point2 in list2:
            distance = euclidean_distance(point1, point2)
            if distance < min_distance:
                min_distance = distance
        distances.append(min_distance)
    return distances


def color_nearest_pixels(image, sample_collection, color, radius=1, white_only=False):

    # Color the nearest pixels around each point
    for sample in sample_collection.list:
        for point in sample.list:
            for y in range(point.y - radius, point.y + radius + 1):
                for x in range(point.x - radius, point.x + radius + 1):
                    # Check if the point is within the image bounds
                    if x >= 0 and x < image.shape[1] and y >= 0 and y < image.shape[0]:
                        #if white_only and image[y, x] == 255:
                        image[y, x] = color
                        #if not white_only and image[y, x] != 255:
                        #    image[y, x] = color
    
    return image


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

    

def decode_color(color):

    white = [0, 0, 0]

    if color.all() == 0:
        print("white!")

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

# Example usage


def display_result(image, display_size):
    # Display the image

    resized_image = cv2.resize(image, (display_size, display_size))
    cv2.imshow('Image', resized_image)

    # Wait for a key press and close the image window
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def min_max_normalize(data):
    # Calculate the minimum and maximum values in the list
    min_val = min(data)
    max_val = max(data)
    
    # Perform min-max normalization
    normalized_data = [(x - min_val) / (max_val - min_val) for x in data]
    
    return normalized_data


def remap(image):
    map_x = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
    map_y = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
    for i in range(map_x.shape[0]):
        map_x[i,:] = [x for x in range(map_x.shape[1])]
    for j in range(map_y.shape[1]):
        map_y[:,j] = [y for y in range(map_y.shape[0])]

    return cv2.remap(image, map_x, map_y, cv2.INTER_NEAREST)


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

    result = convert_to_bgr(image)
    height, width = image.shape

    for y in range(height):
        for x in range(width):
            g_val = image[y, x]
            color_val = get_color_info(config=config, greyscale_value=g_val)
            if color_val == None:
                result[y, x] = [0, 160, 255] # Bright orange
            else:
                result[y, x] = color_val['BGR_value']

    return result


# def get_greyscale_value(config):
#     for c in get_color_info(config=config, greyscale_value=)

def convert_to_bgr(image):
    return cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)

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
    sample_positions = get_free_points(image=map, fraction=5)
    laser_sample = sample_visible_objects(map, sample_positions, detection_colors=[0, 2])

    return laser_sample

def generate_map_data(map):
    sample_positions = get_free_points(image=map, fraction=5)
    map_sample = sample_visible_objects(map, sample_positions, detection_colors=[0, 3])

    return map_sample


def color_point_radius(image, point, color):
    cv2.circle(image, (point.x, point.y), 1, color, -1)
    return image

def create_point_list_from_sample_collection(sample_collection):
    # Extract all points from a sample_collection
    all_points = []
    for sample in sample_collection.samples:
        for p in sample.points:
            all_points.append((p.x, p.y))
    
    return all_points


def calculate_sample_score(sample, kd_tree):
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
    
    # Calculate the RMS distance
    if distances:
        rms_distance = np.sqrt(np.mean(np.square(distances)))
    else:
        rms_distance = None  # Default to None if no points are present

    return rms_distance


import cv2
import numpy as np

def color_gaussian_circle(image, laser_sample_collection, radius, max_score):
    """
    Colors a Gaussian circle for each sample in the laser_sample_collection on the image.
    The color is based on a heatmap that transitions from purple-blue (low values) to bright red (high values).

    Args:
        image (numpy.ndarray): The image to draw on.
        laser_sample_collection (sample_collection): The collection of laser samples.
        radius (int): The radius of the Gaussian circle.
        max_score (float): The maximum score for normalization.
    """
    for sample in laser_sample_collection.samples:
        # Normalize the score to a range of 0 to 1
        normalized_score = sample.score / max_score if max_score > 0 else 0

        # Convert normalized score to a heatmap color (purple-blue to red)
        color = (
            int(255 * (1 - normalized_score)),  # Blue channel
            int(0),                             # Green channel
            int(255 * normalized_score)         # Red channel
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
                        image[py, px] = (
                            np.clip(int(image[py, px][0] * (1 - intensity) + color[0] * intensity), 0, 255),
                            np.clip(int(image[py, px][1] * (1 - intensity) + color[1] * intensity), 0, 255),
                            np.clip(int(image[py, px][2] * (1 - intensity) + color[2] * intensity), 0, 255)
                        )
    return image

def main():

    conf = load_config()
    map = load_image()
    laser_sample_collection, map_sample_collection = load_data("edi/laser_sample.json", "edi/map_sample.json", map)
    


    all_map_sample_points = create_point_list_from_sample_collection(map_sample_collection)
    kd_tree = KDTree(all_map_sample_points)

    scores = []
    for sample in laser_sample_collection.samples:
        sample.score = calculate_sample_score(sample, kd_tree) # Add score to sample
        scores.append(sample.score) # Collect scores in a list

    # Normalize the scores
    normalized_scores = min_max_normalize(scores)


    heat_map = convert_to_bgr(map)
    
    color_gaussian_circle(map, laser_sample_collection, radius=5, max_score=max(normalized_scores))


    # 

    # distance, index = kd_tree.query(p1.to_tuple())
    # print(f"Nearest neighbor to {p1.to_tuple()} is {all_map_sample_points[index]} with distance {distance}")


    #data = sample_collection_pair(laser_sample_collection, map_sample_collection)

    # print(len(laser_sample_collection.samples))
    # print(len(map_sample_collection.samples))

    # for s in laser_sample_collection.samples:
    #     calculate_score(s)




    for s in laser_sample_collection.samples:
        color_point_radius(map, s.origin, 6)

    for s in map_sample_collection.samples:
        for p in s.points:
            color_point_radius(map, p, 5)

    for s in laser_sample_collection.samples:
        for p in s.points:
            color_point_radius(map, p, 4)



    #display_image = get_display_image(heat_map, conf)

    # Serializing json
    

    # with open("sample.json", "w") as outfile:
    #     json.dump(map_sample.to_dict(), outfile, indent=4) 

    save_data(laser_sample_collection, "laser_sample.json")
    save_data(map_sample_collection, "map_sample.json")

    display_result(heat_map, conf["parameters"]["display_window_size"])

    # -----------------------------------------------------------------------
    exit()


    # distances = nearest_neighbor_distances(laser_points, map_sample_points)
    # score = sum(distances)
    # scores.append(score)

    # normalized_scores = min_max_normalize(scores)

    # for i in range(len(normalized_scores)):
    #     free_points[i].score = normalized_scores[i]

    # for p in free_points:
    #     # image[p.y, p.x] = [p.score, p.score, p.score]
    #     image = color_nearest_pixels(image, [[p.y, p.x]], (255, 255-(p.score*255), 0), radius=round(256/fraction), white_only=True)





# Using the special variable 
# __name__
if __name__=="__main__":
    main()