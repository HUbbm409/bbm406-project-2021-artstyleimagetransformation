### Main code for clustering ####
import os, shutil, glob, os.path
import clustering_vgg19 

# Variables
imdir = './vangogh/vangogh/' 
clustered_images_dir = "./clustered_data/vgg_19_style/" 
number_clusters = 10



#### VGG19 STYLE BASED ####
filelist = glob.glob(os.path.join(imdir, '*.jpg'))
clustering_vgg19.feature_
_, all_styles  = clustering_vgg19.vgg19_feature_extracting_style_based(imdir)
kmeans_model_stylebased_vgg19 = clustering_vgg19.k_means(all_styles)


indices0= clustering_vgg19.get_cluster_label_indices(kmeans_model_stylebased_vgg19.labels_,0)
indices1= clustering_vgg19.get_cluster_label_indices(kmeans_model_stylebased_vgg19.labels_,1)
indices2= clustering_vgg19.get_cluster_label_indices(kmeans_model_stylebased_vgg19.labels_,2)


clustering_vgg19.copy_clustered_files(kmeans_model_stylebased_vgg19, clustered_images_dir, filelist)