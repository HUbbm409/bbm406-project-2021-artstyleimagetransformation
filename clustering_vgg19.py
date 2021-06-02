# Imports
from keras.preprocessing import image
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input


from sklearn.cluster import AffinityPropagation
from sklearn import metrics

import numpy as np
from sklearn.cluster import KMeans
import os, shutil, glob, os.path
from PIL import Image as pil_image
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 

from keras import Model
import random

image.LOAD_TRUNCATED_IMAGES = True 
model = VGG19(weights='imagenet', include_top=False)

# Variables


style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1', 
                'block5_conv1']
content_layers = ['block5_conv2'] 

def gram_matrix(input_tensor):
  result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
  input_shape = tf.shape(input_tensor)
  num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
  return result/(num_locations)

def vgg_layers(layer_names):
  """ Creates a vgg model that returns a list of intermediate output values."""
  # Load our model. Load pretrained VGG, trained on imagenet data
  vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
  vgg.trainable = False
  outputs = [vgg.get_layer(name).output for name in layer_names]
  model = tf.keras.Model([vgg.input], outputs)
  return model


class StyleContentModel(tf.keras.models.Model):
  def __init__(self, style_layers, content_layers):
    super(StyleContentModel, self).__init__()
    self.vgg = vgg_layers(style_layers + content_layers)
    self.style_layers = style_layers
    self.content_layers = content_layers
    self.num_style_layers = len(style_layers)
    self.vgg.trainable = False

  def call(self, inputs):
    "Expects float input in [0,1]"
    inputs = inputs*255.0
    preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
    outputs = self.vgg(preprocessed_input)
    style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                      outputs[self.num_style_layers:])
  
    style_outputs = [gram_matrix(style_output)
                     for style_output in style_outputs]

    content_dict = {content_name: value
                    for content_name, value
                    in zip(self.content_layers, content_outputs)}

    style_dict = {style_name: value
                  for style_name, value
                  in zip(self.style_layers, style_outputs)}

    return {'content': content_dict, 'style': style_dict}




''' You can use these methods to extract features, evaluate clustering and use k means algorithm'''

def vgg19_feature_extracting_style_based(imdir):
    # Loop over files and get features
    filelist = glob.glob(os.path.join(imdir, '*.jpg'))
    filelist.sort()
    random.shuffle(filelist)
    featurelist_style_based = {}
    for i, imagepath in enumerate(filelist):
             
            print("    Status: %s / %s" %(i, len(filelist)), end="\r")
            img = image.load_img(imagepath, target_size=(224, 224))
            img_data = image.img_to_array(img)
            img_data = np.expand_dims(img_data, axis=0)
    
            extractor = StyleContentModel(style_layers, content_layers)
            results = extractor(tf.constant(img_data))
    
            for name, output in sorted(results['style'].items()):
                if(name not in list(featurelist_style_based.keys())):
                    featurelist_style_based[name]=[]
                    print(name)
                
                features = output.numpy()
                featurelist_style_based[name].append(features.flatten())
                
    all_styless = np.concatenate((featurelist_style_based['block1_conv1'], featurelist_style_based['block2_conv1']), axis=1)
    all_styless = np.concatenate((all_styless, featurelist_style_based['block3_conv1']), axis=1)
    all_styless = np.concatenate((all_styless, featurelist_style_based['block4_conv1']), axis=1)
    all_styless = np.concatenate((all_styless, featurelist_style_based['block5_conv1']), axis=1)
    return featurelist_style_based, all_styless
            
def get_cluster_label_indices(labels_, label_number):
    return [index_of_label for index_of_label, label in enumerate(labels_) if label == label_number]
            
def vgg19_feature_extracting_object_based(imdir):
    # Loop over files and get features
    # Loop over files and get features
    filelist = glob.glob(os.path.join(imdir, '*.jpg'))
    filelist.sort()
    featurelist = []
    for i, imagepath in enumerate(filelist):
        try:
            print("    Status: %s / %s" %(i, len(filelist)), end="\r")
            img = image.load_img(imagepath, target_size=(224, 224))
            img_data = image.img_to_array(img)
            img_data = np.expand_dims(img_data, axis=0)
            img_data = preprocess_input(img_data)
            features = np.array(model.predict(img_data))
            featurelist.append(features.flatten())
        except:
            continue
    return featurelist
        
        
def elbow_method(featurelist):
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, random_state=0).fit(np.array(featurelist))
        wcss.append(kmeans.inertia_)
    plt.plot(range(1, 11), wcss, 'bx-')
    plt.title('The elbow method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show() 
    


# Compute K Means
def k_means(features, k=3):
    kmeans = KMeans(n_clusters=k, random_state=0).fit(np.array(features))
    return kmeans


# Compute Affinity Propagation
def affinity_propagation(features):
    af = AffinityPropagation(preference=-50).fit(features)
    cluster_centers_indices = af.cluster_centers_indices_
    labels = af.labels_
    return af, cluster_centers_indices

def print_clusters(kmeans):
    import collections
    print("number of dataset in clusters:")
    collections.Counter(kmeans.labels_)
    
def copy_clustered_files(kmeans, targetdir, filelist):
    # Copy images renamed by cluster 
    # Check if target dir exists
    try:
        os.makedirs(targetdir)
    except OSError:
        pass
    # Copy with cluster name
    print("\n")
    for i, m in enumerate(kmeans.labels_):
      cluster_dir=targetdir +"Cluster"+str(m)+"/"+"Cluster"+str(m)+"/"
      try:
        os.makedirs(cluster_dir)
      except OSError:
          print("    Copy: %s / %s" %(i, len(kmeans.labels_)), end="\r")
          shutil.copy(filelist[i], cluster_dir+"Cluster"+ str(m) + "_" + str(i) + ".jpg")
          
          
          
def save_clustered_filenames(filelist, indices2, filename="file_cluster_2.csv"):
    file_cluster_2=[]
    for i in range(1, len(indices2)):
        file_cluster_2.append(filelist[indices2[i]])
        
    
    import numpy as np
    np.savetxt(filename, 
               file_cluster_2,
               delimiter =", ", 
               fmt ='% s')