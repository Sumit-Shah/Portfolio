import os
import pickle
import numpy as np
import cv2
import faiss
import matplotlib.pyplot as plt
from tensorflow.keras.applications import ResNet50, InceptionV3, EfficientNetB0
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_inception
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_efficientnet
from tensorflow.keras.preprocessing import image


# Loading the models and preprocessing  the functions.
# Loading the pre-trained models and preprocess functions.

# Load pre-trained models
models = {
    'resnet': ResNet50(weights='imagenet', include_top=False, pooling='avg'),
    'inception': InceptionV3(weights='imagenet', include_top=False, pooling='avg'),
    'efficientnet': EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')
}

# Corresponding preprocess functions
preprocess_funcs = {
    'resnet': preprocess_resnet,
    'inception': preprocess_inception,
    'efficientnet': preprocess_efficientnet
}


# Image Processing Function.
def enhance_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

def preprocess_image(img_path, model_name, enhance=False):
    img = enhance_image(img_path) if enhance else cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    target_size = (299, 299) if model_name == 'inception' else (224, 224)
    img = cv2.resize(img, target_size)
    img_array = np.expand_dims(img, axis=0)
    return preprocess_funcs[model_name](img_array)

def get_embedding(img_path, model_name, enhance=False):
    img_array = preprocess_image(img_path, model_name, enhance)
    model = models[model_name]
    embedding = model.predict(img_array).flatten()
    norm = np.linalg.norm(embedding)
    return embedding / norm if norm != 0 else embedding

def get_aggregated_embedding(img_path, enhance=False):
    embeddings = [get_embedding(img_path, model_name, enhance) for model_name in models]
    combined_embedding = np.concatenate(embeddings)
    norm = np.linalg.norm(combined_embedding)
    return combined_embedding / norm if norm != 0 else combined_embedding


# FAISS Index MAnagment
def create_faiss_index(embeddings, image_ids, index_path='faiss_index.index'):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    faiss.write_index(index, index_path)
    with open('id_mapping.pkl', 'wb') as f:
        pickle.dump(image_ids, f)

def load_faiss_index(index_path='faiss_index.index'):
    return faiss.read_index(index_path)

def load_id_mapping(mapping_path='id_mapping.pkl'):
    with open(mapping_path, 'rb') as f:
        return pickle.load(f)

def search_faiss_index(index, query_embedding, k=1):
    distances, indices = index.search(query_embedding.reshape(1, -1), k)
    return distances, indices



# Database and Embedding Storage
import sqlite3

def initialize_db(db_path='image_embeddings.db'):
    with sqlite3.connect(db_path) as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS embeddings (
                image_id TEXT PRIMARY KEY,
                embedding BLOB
            )
        ''')

def store_embedding(img_name, embedding, db_path='image_embeddings.db'):
    with sqlite3.connect(db_path) as conn:
        conn.execute('''
            INSERT OR REPLACE INTO embeddings (image_id, embedding)
            VALUES (?, ?)
        ''', (img_name, pickle.dumps(embedding)))

def retrieve_embedding(img_name, db_path='image_embeddings.db'):
    with sqlite3.connect(db_path) as conn:
        result = conn.execute('''
            SELECT embedding FROM embeddings WHERE image_id=?
        ''', (img_name,)).fetchone()
    return pickle.loads(result[0]) if result else None

def store_embeddings(folder_path, db_path='image_embeddings.db'):
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Directory not found: {folder_path}")

    embeddings = []
    image_ids = []

    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        if os.path.isfile(img_path):
            try:
                embedding = get_aggregated_embedding(img_path)
                store_embedding(img_name, embedding, db_path)
                embeddings.append(embedding)
                image_ids.append(img_name)
                print(f"Stored embedding for: {img_name}")
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

    if embeddings:
        create_faiss_index(np.array(embeddings), image_ids)

# Matching and Display Functions
def find_matching_image(new_img_path, db_path='image_embeddings.db', index_path='faiss_index.index', k=1, enhance=False):
    try:
        new_img_embedding = get_aggregated_embedding(new_img_path, enhance)
    except Exception as e:
        print(f"Error processing new image {new_img_path}: {e}")
        return None, 0

    index = load_faiss_index(index_path)
    id_mapping = load_id_mapping()

    distances, indices = search_faiss_index(index, new_img_embedding, k)
    if indices.size == 0:
        return None, 0

    matched_id = id_mapping[int(indices[0][0])]
    similarity = 1 / (1 + distances[0][0])
    return matched_id, similarity

def display_image(img_path, title="Image"):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Image not found: {img_path}")
        return
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

# Execution

# Paths
image_folder = '/content/drive/MyDrive/Python_Image_Embedding/zimisa_collection'
single_image_paths = [
    '/content/drive/MyDrive/Python_Image_Embedding/65fd6ce46c776.jpg.webp',
    '/content/drive/MyDrive/Python_Image_Embedding/6444e5745ccbe.jpg',
    '/content/drive/MyDrive/Python_Image_Embedding/ef9bef967ebf43f7997b4fe03ee7dfea.jpg_2200x2200q80.jpg_.webp',
    '/content/drive/MyDrive/Python_Image_Embedding/S6e93164efdfb4145ba64afb1dc4390524.jpg'
]

# Process each image
for img_path in single_image_paths:
    # Enhance the image
    enhanced_img = enhance_image(img_path)

    # Save enhanced image temporarily
    enhanced_img_path = img_path.replace('.jpg', '_enhanced.jpg')
    cv2.imwrite(enhanced_img_path, enhanced_img)

    # Find and display matches for the enhanced image
    enhance = True
    matched_img, similarity = find_matching_image(enhanced_img_path, db_path='/content/drive/MyDrive/Python_Image_Embedding/image_embeddings.db', enhance=enhance)

    if matched_img:
        matched_img_path = os.path.join(image_folder, matched_img)
        if os.path.isfile(matched_img_path):
            print(f"Match found for {img_path}: {matched_img} with similarity: {similarity:.2f}")
            display_image(matched_img_path, title=f"Matched Image: {os.path.basename(matched_img_path)}")
        else:
            print(f"File {matched_img_path} does not exist.")
    else:
        print(f"No matching image found for {img_path}.")