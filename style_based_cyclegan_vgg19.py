!pip install git+https://github.com/tensorflow/examples.git

import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix

import os
import time
import matplotlib.pyplot as plt
from IPython.display import clear_output
import numpy as np


AUTOTUNE = tf.data.AUTOTUNE


BUFFER_SIZE = 1000
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256



'''##Loading Files, if full data will be used, ##'''

clustered_images_dir = "./clustered_data/vgg_19_style/" 
train_vangogh_ds = tf.keras.preprocessing.image_dataset_from_directory(targetdir+"Cluster1",
    image_size=(256, 256),
    seed=1337,
    batch_size = 1,
    interpolation='nearest' 
 )

train_normal_ds = tf.keras.preprocessing.image_dataset_from_directory("./normal/",
    image_size=(256, 256),
    seed=1337,
    batch_size = 1,
    interpolation='nearest' 
 )

test_normal_ds = tf.keras.preprocessing.image_dataset_from_directory("./normal-test/",
    image_size=(256, 256),
    seed=1337,
    batch_size = 1,
    interpolation='nearest' 
 )



'''### DATA AUGMENTATION ###'''
def random_crop(image):
  cropped_image = tf.image.random_crop(
      image, size=[IMG_HEIGHT, IMG_WIDTH, 3])

  return cropped_image


# normalizing the images to [-1, 1]
def normalize(image):
  image = tf.cast(image, tf.float32)
  image = (image / 127.5) - 1
  return image

def random_jitter(image):
  # resizing to 286 x 286 x 3
  #image = tf.image.resize(image, [286, 286], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  # randomly cropping to 256 x 256 x 3
  image = random_crop(image)

  # random mirroring
  image = tf.image.random_flip_left_right(image)

  return image

def preprocess_image_train(image, label):
  print(image)
  image = random_jitter(image[0])
  print(image)
  image = normalize(image)
  print(image)

  return image

def preprocess_image_test(image, label):
  image = normalize(image)
  return image



'''## PREPROCESSING IMAGES ##'''
train_vangogh_ds_prep = train_vangogh_ds.map( preprocess_image_train, num_parallel_calls=AUTOTUNE).cache().shuffle( BUFFER_SIZE).batch(BATCH_SIZE)
train_normal_ds_prep = train_normal_ds.map( preprocess_image_train, num_parallel_calls=AUTOTUNE).cache().shuffle( BUFFER_SIZE).batch(BATCH_SIZE)
test_normal_ds_prep = test_normal_ds.map( preprocess_image_train, num_parallel_calls=AUTOTUNE).cache().shuffle( BUFFER_SIZE).batch(BATCH_SIZE)




'''## UNET GENERATOR ##'''
OUTPUT_CHANNELS = 3

generator_g = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
generator_f = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')

discriminator_x = pix2pix.discriminator(norm_type='instancenorm', target=False)
discriminator_y = pix2pix.discriminator(norm_type='instancenorm', target=False)





'''## LOSS FUNCTIONS ##'''
LAMBDA = 10
loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)
def discriminator_loss(real, generated):
  real_loss = loss_obj(tf.ones_like(real), real)

  generated_loss = loss_obj(tf.zeros_like(generated), generated)

  total_disc_loss = real_loss + generated_loss

  return total_disc_loss * 0.5

def generator_loss(generated):
  return loss_obj(tf.ones_like(generated), generated)

def calc_cycle_loss(real_image, cycled_image):
  loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))

  return LAMBDA * loss1

def identity_loss(real_image, same_image):
  loss = tf.reduce_mean(tf.abs(real_image - same_image))
  return LAMBDA * 0.5 * loss





'''## ADAM OPTIMIZER ##'''
generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)





'''## GENERATING IMAGE ##'''
def generate_images(model, test_input):
  prediction = model(test_input)

  plt.figure(figsize=(12, 12))

  display_list = [test_input[0], prediction[0]]
  title = ['Input Image', 'Predicted Image']

  for i in range(2):
    plt.subplot(1, 2, i+1)
    plt.title(title[i])
    # getting the pixel values between [0, 1] to plot it.
    plt.imshow(display_list[i] * 0.5 + 0.5)
    plt.axis('off')
  plt.show()
  
  
  
  
  
'''### TRAIN STEP ###'''
def train_step(real_x, real_y):
  # persistent is set to True because the tape is used more than
  # once to calculate the gradients.
  with tf.GradientTape(persistent=True) as tape:
    # Generator G translates X -> Y
    # Generator F translates Y -> X.

    fake_y = generator_g(real_x, training=True)
    cycled_x = generator_f(fake_y, training=True)

    fake_x = generator_f(real_y, training=True)
    cycled_y = generator_g(fake_x, training=True)

    # same_x and same_y are used for identity loss.
    same_x = generator_f(real_x, training=True)
    same_y = generator_g(real_y, training=True)

    disc_real_x = discriminator_x(real_x, training=True)
    disc_real_y = discriminator_y(real_y, training=True)

    disc_fake_x = discriminator_x(fake_x, training=True)
    disc_fake_y = discriminator_y(fake_y, training=True)

    # calculate the loss
    gen_g_loss = generator_loss(disc_fake_y)
    gen_f_loss = generator_loss(disc_fake_x)

    total_cycle_loss = calc_cycle_loss(real_x, cycled_x) + calc_cycle_loss(real_y, cycled_y)

    print("Total: ", total_cycle_loss)

    # Total generator loss = adversarial loss + cycle loss
    id_loss_y = identity_loss(real_y, same_y)
    id_loss_x = identity_loss(real_x, same_x)
    total_gen_g_loss = gen_g_loss + total_cycle_loss + id_loss_y
    total_gen_f_loss = gen_f_loss + total_cycle_loss + id_loss_x

    disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
    disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)

  # Calculate the gradients for generator and discriminator
  generator_g_gradients = tape.gradient(total_gen_g_loss, 
                                        generator_g.trainable_variables)
  generator_f_gradients = tape.gradient(total_gen_f_loss, 
                                        generator_f.trainable_variables)

  discriminator_x_gradients = tape.gradient(disc_x_loss, 
                                            discriminator_x.trainable_variables)
  discriminator_y_gradients = tape.gradient(disc_y_loss, 
                                            discriminator_y.trainable_variables)

  # Apply the gradients to the optimizer
  generator_g_optimizer.apply_gradients(zip(generator_g_gradients, 
                                            generator_g.trainable_variables))

  generator_f_optimizer.apply_gradients(zip(generator_f_gradients, 
                                            generator_f.trainable_variables))

  discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients,
                                                discriminator_x.trainable_variables))

  discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients,
                                                discriminator_y.trainable_variables))
  return (total_cycle_loss, gen_g_loss, gen_f_loss, disc_x_loss, disc_y_loss, id_loss_x, id_loss_y)







''' ### TRAINING PART### '''
ex_for_test = next(iter(test_normal_ds_prep))

cycle_loss_list = []
gen_g_loss_list = []
gen_f_loss_list = []
disc_x_loss_list = []
disc_y_loss_list = []
id_loss_x_list = []
id_loss_y_list = []

EPOCHS = 50

for epoch in range(EPOCHS):
  start = time.time()

  n = 0
  total_cycle_loss = 0
  total_gen_g_loss_loss = 0
  total_gen_f_loss_loss = 0
  total_disc_x_loss_loss = 0
  total_disc_y_loss_loss = 0
  total_id_loss_x_loss = 0
  total_id_loss_y_loss = 0
  length=0
  for image_x, image_y in tf.data.Dataset.zip((train_normal_ds_prep, train_vangogh_ds_prep)):
    cycle_loss, gen_g_loss, gen_f_loss, disc_x_loss, disc_y_loss, id_loss_x, id_loss_y = train_step_resnet(image_x, image_y)
    total_cycle_loss += cycle_loss
    total_gen_g_loss_loss += gen_g_loss
    total_gen_f_loss_loss += gen_f_loss
    total_disc_x_loss_loss += disc_x_loss
    total_disc_y_loss_loss += disc_y_loss
    total_id_loss_x_loss += id_loss_x
    total_id_loss_y_loss += id_loss_y
    length+=1
    if n % 10 == 0:
      print ('.', end='')
    n += 1

  cycle_loss_list.append(total_cycle_loss/length)
  gen_g_loss_list.append(total_gen_g_loss_loss/length)
  gen_f_loss_list.append(total_gen_f_loss_loss/length)
  disc_x_loss_list.append(total_disc_x_loss_loss/length)
  disc_y_loss_list.append(total_disc_y_loss_loss/length)
  id_loss_x_list.append(total_id_loss_x_loss/length)
  id_loss_y_list.append(total_id_loss_y_loss/length)

  clear_output(wait=True)
  # Using a consistent image (sample_normal) so that the progress of the model
  # is clearly visible.
  generate_images(generator_resnet_g, processed_deniz_img)

  if (epoch + 1) % 5 == 0:
    ckpt_save_path = ckpt_manager.save()
    print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                         ckpt_save_path))

  print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                      time.time()-start))
  
  



'''### PLOTTING LOSS VALUES###'''
import pandas as pd
import numpy as np

df_loss = pd.DataFrame(columns=["cycle_loss","gen_g_loss","gen_f_loss","disc_x_loss","disc_y_loss","id_loss_x","id_loss_y"])
df_loss["cycle_loss"] = pd.Series(np.array(cycle_loss_list))
df_loss["gen_g_loss"] = pd.Series(np.array(gen_g_loss_list))
df_loss["gen_f_loss"] = pd.Series(np.array(gen_f_loss_list))
df_loss["disc_x_loss"] = pd.Series(np.array(disc_x_loss_list))
df_loss["disc_y_loss"] = pd.Series(np.array(disc_y_loss_list))
df_loss["id_loss_x"] = pd.Series(np.array(id_loss_x_list))
df_loss["id_loss_y"] = pd.Series(np.array(id_loss_y_list))




'''SAVING LOSS VALUES'''
import numpy as np
np.savetxt("cycle_loss_list", cycle_loss_list, delimiter =", ", fmt ='% s')
np.savetxt("gen_g_loss_list", gen_g_loss_list, delimiter =", ", fmt ='% s')
np.savetxt("gen_f_loss_list", gen_f_loss_list, delimiter =", ", fmt ='% s')
np.savetxt("disc_x_loss_list", disc_x_loss_list, delimiter =", ", fmt ='% s')
np.savetxt("disc_y_loss_list", disc_y_loss_list, delimiter =", ", fmt ='% s')
np.savetxt("id_loss_x_list", id_loss_x_list, delimiter =", ", fmt ='% s')
np.savetxt("id_loss_y_list", id_loss_y_list, delimiter =", ", fmt ='% s')