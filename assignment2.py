import numpy as np
from collections import defaultdict
import cv2
import faiss
import struct

def build_kdtree(dataset, max_descriptors=400, num_trees=4):
    """
    Builds a FLANN-based KD-Tree index for fast nearest neighbor search over image descriptors.

    This function collects database descriptors from all images in the provided dataset, 
    limits the number of descriptors per image to `max_descriptors`, and builds 
    the KD-Tree using OpenCV's FLANN (Fast Library for Approximate Nearest Neighbors).

    Parameters:
        - dataset: An instance of `HolidaysDatasetHandler`, which should have methods 
          `get_database_images()` and `get_descriptors(image_name)` to retrieve image 
          names and their corresponding descriptors.
        - max_descriptors: Maximum number of descriptors to consider per image. 
          Default is 400.
        - num_trees: Number of trees in the FLANN KD-Tree. Default is 4.
    
    Returns:
        - flann: The built FLANN-based KD-Tree index for fast nearest neighbor search.
        - image_map: A list that maps the index of each image to its name.
    """
    
    # Initialize the FLANN matcher (will be populated later)
    flann = None   
    image_map = []  # List to map imgIdx (DMatch) to image names
    
    # YOUR CODE HERE
    descriptors_list = []
    for image_name in dataset.get_database_images():
        descriptors = dataset.get_descriptors(image_name)
        if descriptors is not None:
            if descriptors.shape[0] > max_descriptors:
                descriptors = descriptors[:max_descriptors]
            descriptors_list.append(descriptors)
            image_map.append(image_name)
    index_params = dict(algorithm = 1, trees = num_trees)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params) # type: ignore
    if descriptors_list:
            flann.add(descriptors_list)
            flann.train()
    # -----

    return flann, image_map


def search_kdtree(query_descs, flann, k=2, ratio_test=0.75):
    """
    Search the KD-Tree for the best matching images for the query descriptors using FLANN-based matching.

    This function performs a nearest neighbor search using FLANN on the provided query descriptors,
    and applies Lowe's ratio test to filter weak matches. It returns a list of image indices ranked by
    the number of matches found.

    Parameters:
        - query_descs: The descriptors of the query image (should be a NumPy array of shape (N, 128)).
        - flann: The FLANN index built using `build_kdtree`, which holds the image descriptors for fast search.
        - k: Number of nearest neighbors to retrieve for each descriptor. Default is 2.
        - ratio_test: The threshold for Loweâ€™s ratio test, which is used to filter weak matches. Default is 0.75.

    Returns:
        - ranked_image_ids: A list of image indices, ranked in descending order of the number of good matches.
          The image with the most good matches is ranked first.
    """

    # YOUR CODE HERE
    ranked_image_ids = []
    matches = flann.knnMatch(query_descs,k=k)
    matches_counts = defaultdict(int)
    for (m,n) in matches:
        if(m.distance < ratio_test*n.distance):
            ranked_image_ids.append(m.imgIdx)
    
    return ranked_image_ids
    # -----


def build_lsh(dataset, max_descriptors=400, num_bits=128):
    """
    Builds an LSH index using FAISS for fast approximate nearest neighbor search.

    Parameters:
        - dataset: Instance of `HolidaysDatasetHandler` to retrieve image names and descriptors.
        - max_descriptors: Maximum descriptors per image (default: 400).
        - num_bits: Number of bits for LSH hashing (default: 128, matching SIFT descriptor size).

    Returns:
        - lsh_index: The FAISS LSH index.
        - image_map: List mapping each feature's index in FAISS to its corresponding image name.
    """

    # Initialize the FAISS index (will be populated later)
    lsh_index = None
    image_map = []  # List to map features to image names
    
    # YOUR CODE HERE

    database_images = dataset.get_database_images()

    d = 128

    lsh_index = faiss.IndexLSH(d, num_bits)

    for image in database_images:
      curr_descriptors = dataset.get_descriptors(image)[:max_descriptors]
      if curr_descriptors is None:
        continue
      lsh_index.add(curr_descriptors)
      image_map.extend([image] * len(curr_descriptors))

    # -----

    return lsh_index, image_map


def search_lsh(query_descs_dict, lsh_index, image_map, k=2):
    """
    Searches the LSH index for the best matching images for multiple query images at once.

    Parameters:
        - query_descs_dict: Dictionary where keys are query image names and values are 
          NumPy arrays of shape (N, 128) representing descriptors.
        - lsh_index: FAISS LSH index built with `build_lsh_index()`.
        - image_map: List mapping feature index in FAISS to its corresponding image.
        - k: Number of nearest neighbors to retrieve per descriptor.

    Returns:
        - ranked_dict: Dictionary mapping query image names to ranked lists of matching images.
    """

    ranked_dict = {}

    # YOUR CODE HERE

    for filename, descriptors in query_descs_dict.items():
        if descriptors is None:
            continue

        distances, indices = lsh_index.search(descriptors, k) 

        votes = Counter()
        for idx in indices.flatten():
            if idx == -1:
                continue
            image_name = image_map[idx]
            votes[image_name] += 1              

        ranked_dict[filename] = [img for img, _ in votes.most_common()]
        
    # -----

    return ranked_dict


class BagOfWordsRetriever:
    """
    A Bag-of-Words (BoW) based image retrieval system using a visual vocabulary.

    This class:
    - Quantizes local descriptors using a precomputed vocabulary.
    - Computes BoW histograms for images.
    - Applies TF-IDF weighting to enhance retrieval accuracy.
    - Uses an inverted file index for fast candidate selection.
    - Retrieves images based on cosine similarity.

    Attributes:
        dataset_handler (HolidaysDatasetHandler): Handles image data and descriptors.
        vocabulary (np.ndarray): The visual vocabulary (loaded from a .fvecs file).
        vocindex (faiss.IndexFlatL2): FAISS Flat L2 for nearest-neighbor search.
        image_histograms (dict): Stores BoW histograms for each image.
        inverted_index (dict): Maps visual words to images containing them.
        use_tfidf (bool): Flag indicating whether TF-IDF weighting is applied.
    """

    def __init__(self, dataset_handler, vocabulary_file):
        """
        Initializes the BoW retrieval system.

        Args:
            dataset_handler (HolidaysDatasetHandler): The dataset handler instance.
            vocabulary_file (str): Path to the vocabulary file (.fvecs format).
        """
        self.dataset_handler = dataset_handler
        self.vocabulary = self.load_vocabulary(vocabulary_file)

        # Build a FAISS L2 index for efficient nearest neighbor search
        self.vocindex = faiss.IndexFlatL2(self.vocabulary.shape[1])
        self.vocindex.add(self.vocabulary)

        # Storage for image histograms and inverted index
        self.image_histograms = {}
        self.inverted_index = {}

        # Flag to indicate if TF-IDF has been applied
        self.use_tfidf = False

    def load_vocabulary(self, filename):
        """
        Loads the visual vocabulary from a .fvecs file.

        Args:
            filename (str): Path to the vocabulary file.

        Returns:
            np.ndarray: The loaded visual vocabulary.
        """
        centroids = []

        with open(filename, "rb") as f:
            while True:
                dim_data = f.read(4)
                if not dim_data:
                    break
    
                desdim = struct.unpack("<i", dim_data)[0]
    
                if desdim != 128:
                    raise ValueError(f"Unexpected descriptor dimension {desdim} in {file_path}")
    
                components = np.frombuffer(f.read(128 * 4), dtype=np.float32)
                centroids.append(components)
    
        return np.array(centroids)

    def compute_bow_representation(self, image_name, descriptors):
        """
        Computes the Bag-of-Words (BoW) histogram for an image.

        Steps:
        1. Assigns each descriptor to the closest visual word.
        2. Builds a histogram of visual word occurrences and saves it
        3. Updates the inverted index for fast retrieval.

        Args:
            image_name (str): Name of the image.
            descriptors (np.ndarray): Local descriptors extracted from the image.
        """
        
        # YOUR CODE HERE
        raise NotImplementedError()
        # -----

    def apply_tfidf(self):
        """
        Applies Term Frequency - Inverse Document Frequency (TF-IDF) weighting
        to all image histograms to enhance retrieval performance.

        TF = (count of word in image) / (total words in image)  
        IDF = log( (total images + 1) / (number of images containing the word + 1) ) + 1  
        TF-IDF = TF * IDF

        Modifies:
            - `self.image_histograms`: Stores weighted histograms.
            - `self.use_tfidf`: Marks TF-IDF as applied.
        """

        # YOUR CODE HERE
        raise NotImplementedError()
        # -----

    def retrieve_images(self, query_image, top_k=10):
        """
        Retrieves the most similar images to a given query image.

        Steps:
        1. Computes the BoW histogram for the query image.
        2. If TF-IDF is enabled, applies it to the query.
        3. Uses the inverted index to find relevant images efficiently.
        4. Ranks candidates using cosine similarity.

        Args:
            query_image (str): The query image filename.
            top_k (int): Number of top similar images to retrieve.

        Returns:
            List[Tuple[str, float]]: Top-K retrieved images and similarity scores.
        """

        # YOUR CODE HERE
        raise NotImplementedError()
        # -----


def build_hnsw(dataset_handler, deep_features, M=32, efConstruction=64):
    """
    Builds an HNSW index using FAISS with the provided deep features.

    This function retrieves the database image names from the dataset handler,
    extracts the corresponding vectors from the deep_features dictionary, 
    normalizes them, and builds an HNSW index.

    Parameters:
        - dataset_handler (HolidaysDatasetHandler): The dataset handler instance containing the database image list.
        - deep_features (dict): Dictionary mapping image names to feature vectors.
        - M (int): Number of connections per node in the HNSW graph.
        - efConstruction (int): Depth of search during graph construction.

    Returns:
        - hnsw_index (faiss.IndexHNSWFlat): The built and trained HNSW index.
        - image_map (list): List mapping each feature's index in FAISS to its corresponding image name.
    """

     # Initialize the FAISS index (will be populated later)
    hnsw_index = None
    image_map = []  # List to map features to image names
    
    # YOUR CODE HERE
    raise NotImplementedError()
    # -----
    
    return hnsw_index, image_map
    

def search_hnsw(query_descs_dict, index, image_map, k=10, efSearch=32):
    """
    Searches the HNSW index for the best matching images for multiple query images at once.

    Parameters:
        - query_descs_dict: Dictionary where keys are query image names and values are
           NumPy arrays representing descriptors.
        - index: FAISS HNSW index built with `build_hnsw()`.
        - image_map: List mapping feature index in FAISS to its corresponding image.
        - k: Number of nearest neighbors to retrieve per descriptor.
        - efSearch: Depth of search at query time.

    Returns:
        - ranked_dict: Dictionary mapping query image names to ranked lists of matching images.
    """

    ranked_dict = {}

    # YOUR CODE HERE
    raise NotImplementedError()
    # -----

    return ranked_dict