import numpy as np
import struct
import os
import cv2
import sys

class HolidaysDatasetHandler:
    def __init__(self, root_dir, load_features=False):
        """
        Initialize the DatasetHolidays class.

        Args:
            root_dir (str): Path to the root directory of the dataset.
            load_features (bool): Whether to load features during initialization.
        """
        self.root_dir = root_dir
        self.images_list_file = os.path.join(root_dir, "holidays_images.dat")
        self.images_dir = os.path.join(root_dir, "images")        
        self.features_dir = os.path.join(root_dir, "features")

        self.data = {}
        self.ground_truth = {}
        self.query_images = []
        self.database_images = []
        self.loaded_features = load_features

        self._load_image_list()
        self._get_groundtruth()      

        if load_features:
            self.load_features()

    def _load_image_list(self):
        """Load the list of images from the dataset file."""
        if not os.path.exists(self.images_list_file):
            raise FileNotFoundError(f"Image list file not found: {self.images_list_file}")

        with open(self.images_list_file, "r") as file:
            for line in file:
                image_name = line.strip()
                self.data[image_name] = {
                    "path": os.path.join(self.images_dir, image_name),
                    "image": None,
                    "image_loaded": False,
                    "keypoints": None,
                    "descriptors": None
                }
                
    def _get_groundtruth(self):
        """Generate ground truth mapping queries to relevant results."""        
        gt = {}
        with open(self.images_list_file, "r") as file:
            for line in file:
                imname = line.strip()
                imno = int(imname[:-len(".jpg")])
                if imno % 100 == 0:
                    gt_results = set()
                    gt[imname] = gt_results
                    self.query_images.append(imname)
                else:
                    gt_results.add(imname)
                    self.database_images.append(imname)

        self.ground_truth = gt

    def get_image(self, image_name):
        """Load images into the dataset dictionary using OpenCV."""
        info = self.data[image_name]
        if os.path.exists(info["path"]) and not info["image_loaded"]:
            image = cv2.imread(info["path"])
            if image is not None:
                info["image"] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                info["image_loaded"] = True
        
        return info["image"]

    def load_features(self):
        """Load features (keypoints and descriptors) for each image."""
        print('Loading features ...');
        for image_name, info in self.data.items():
            feature_file = os.path.join(self.features_dir, image_name.replace(".jpg", ".siftgeo"))
            if os.path.exists(feature_file):
                keypoints, descriptors = self._parse_siftgeo(feature_file)
                info["keypoints"] = keypoints
                info["descriptors"] = descriptors

        self.loaded_features = True
        print('Completed!')
                
    def _parse_siftgeo(self, file_path):
        """
        Parse a .siftgeo file to extract keypoints and descriptors.

        Args:
            file_path (str): Path to the .siftgeo file.

        Returns:
            keypoints (list of dict): List of keypoints with their properties.
            descriptors (numpy.ndarray): Array of descriptors, shape (n, 128).
        """
        keypoints = []
        descriptors = []

        descriptor_size = 168  # Size of one descriptor block in bytes

        with open(file_path, "rb") as f:
            data = f.read()

        num_descriptors = len(data) // descriptor_size

        for i in range(num_descriptors):
            offset = i * descriptor_size
            block = data[offset:offset + descriptor_size]

            # Unpack the keypoint fields
            x, y, scale, angle, mi11, mi12, mi21, mi22, cornerness = struct.unpack("<9f", block[:36])
            desdim = struct.unpack("<i", block[36:40])[0]

            if desdim != 128:
                raise ValueError(f"Unexpected descriptor dimension {desdim} in {file_path}")

            descriptor_vector = np.frombuffer(block[40:40 + desdim], dtype=np.uint8).astype(np.float32)

            keypoints.append(cv2.KeyPoint(x=x, y=y, size=scale, angle=angle, response=cornerness))
            descriptors.append(descriptor_vector)

        descriptors = np.array(descriptors, dtype=np.float32)
        return keypoints, descriptors

    def get_kps(self, image_name):
        """
        Get the keypoints for a given image.
    
        Args:
            image_name (str): The filename of the image.
    
        Returns:
            list or None: A list of keypoints if features are loaded, otherwise None.
        """
        info = self.data[image_name]
        if self.loaded_features:
            return info["keypoints"]
        else:
            return None

    def get_descriptors(self, image_name):
        """
        Get the descriptors for a given image.
    
        Args:
            image_name (str): The filename of the image.
    
        Returns:
            list or None: A list of descriptors if features are loaded, otherwise None.
        """
        info = self.data[image_name]
        if self.loaded_features:
            return info["descriptors"]
        else:
            return None
    
    def get_query_images(self):
        """
        Retrieve the list of query images.
    
        Returns:
            list: A list of filenames representing the query images.
        """
        return self.query_images

    def get_database_images(self):
        """
        Retrieve the list of database images.
    
        Returns:
            list: A list of filenames representing the database images.
        """
        return self.database_images

    def compute_AP(self, image_name, ranked_list):
        """
        Compute the Average Precision (AP) for a given query image.
    
        Args:
            image_name (str): The query image filename.
            ranked_list (list): A list of ranked image filenames.
    
        Returns:
            float: The average precision for the query image, or 0 if no relevant images exist.
    
        Example:
            >>> ap = compute_AP("100000.jpg", ["100001.jpg", "100002.jpg"])
            >>> print(ap)
            1.0
        """

        if image_name not in self.ground_truth:
            print(f"Error: {image_name} not found in ground truth.")
            return

        relevant_imgs = set(self.ground_truth[image_name])  # Ensure this is a set for fast lookup
    
        ap = 0.0
        nrel = len(relevant_imgs)
        
        if nrel == 0:  # If no relevant images, return 0 or handle accordingly
            print(f"No relevant images found for {image_name}.")
            return 0
    
        curr_k = 1
        curr_rel = 0
    
        for imname in ranked_list:
            # Checking if the returning result is relevant to the query
            if imname in relevant_imgs:
                curr_rel += 1
                ap += float(curr_rel) / float(curr_k)
            curr_k += 1
    
        return ap / nrel

    def compute_mAP(self, ranked_dict):
        """
        Compute the Mean Average Precision (mAP) for a set of queries.
        
        Args:
            ranked_dict (dict): A dictionary where keys are query image filenames and values 
                                 are lists of ranked image filenames for each query.
        
        Returns:
            float: The mean average precision for the set of queries.
        
        Example:
            >>> ranked_dict = {
            >>>     "100000.jpg": ["100001.jpg", "100002.jpg"],
            >>>     "100100.jpg": ["100101.jpg"]
            >>> }
            >>> map_score = compute_map(ranked_dict)
            >>> print(map_score)
            1.0
        """
    
        ap_sum = 0.0
        num_queries = len(ranked_dict)
    
        for image_name, ranked_list in ranked_dict.items():
            ap = self.compute_AP(image_name, ranked_list)
            
            if ap is not None:  # Only add AP if it's computed (not None)
                ap_sum += ap
    
        # Return the mean of the average precision values
        return ap_sum / num_queries if num_queries > 0 else 0


# Example usage
if __name__ == "__main__":
    dataset = HolidaysDatasetHandler("holidays", load_features=True)

    # Mostrar una imagen del dataset usando OpenCV
    for image_name, info in dataset.data.items():
        print(f"Mostrando la imagen: {image_name}")
        img = dataset.get_image(image_name)
        #cv2.imshow("Image", img)
        #cv2.waitKey(0)  # Espera a que se presione una tecla para cerrar la ventana
        #cv2.destroyAllWindows()
        break

    kps = dataset.get_kps('100000.jpg')
    print(kps[0].pt[0], ' ', kps[0].pt[1])

    descs = dataset.get_descriptors('100000.jpg')
    print(descs)    
    print(descs.shape)

    ap = dataset.compute_AP('100000.jpg', ["100001.jpg"])
    print(ap)

    ranked_dict = {
        "100000.jpg": ["100001.jpg"],
        "100100.jpg": ["100101.jpg"]
    }    
    map_score = dataset.compute_mAP(ranked_dict)
    print(map_score)

    query_images = dataset.get_query_images()
    print(len(query_images))

    database_images = dataset.get_database_images()
    print(len(database_images))