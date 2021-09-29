# setup (run only if needed)
#install.packages(c("keras","tfdatasets","mapview","stars","rsample","gdalUtils","purrr", "magick", "jpeg"))
#reticulate::install_miniconda()
#keras::install_keras()
#reticulate::py_config()
#tensorflow::tf_config()
#keras::is_keras_available()

# packages
library(keras)
library(tensorflow)
library(tfdatasets)
library(purrr)
library(ggplot2)
library(rsample)
library(stars)
library(raster)
library(reticulate)
#library(mapview)

setwd("C:/Users/Cornelia/Documents/Studium/EAGLE/2Semester")

## DATA PREPROCESSING
# index data

data <- rbind(
  data.frame(
    img = list.files("C:/Users/Cornelia/Documents/Studium/EAGLE/2Semester/AdvancedProgramming/Project/sat_tiles_0_1_29_9_png_3bands", pattern=".png$", full.names = T),
    mask  = list.files("C:/Users/Cornelia/Documents/Studium/EAGLE/2Semester/AdvancedProgramming/Project/mask_tiles_0_1_29_9_png", pattern=".png$", full.names = T)
  )
)



# split of data into training and testing
data <- initial_split(data, prop = 0.75)
data
# defining shape and batch size
input_shape <- c(448,448,3)
batch_size <- 10

# call training and testing data
training(data)
testing(data)

# create tensor slices from our training data
train_ds <- tensor_slices_dataset(training(data))

# load images and masks values into the tensor slices (i.e. fill the tensors)
train_ds <- dataset_map(
  train_ds, function(x) 
    list_modify(x, 
                #img = tf$image$decode_jpeg(tf$io$read_file(x$img)),
                #mask = tf$image$decode_jpeg(tf$io$read_file(x$mask)))
                img = tf$image$decode_png(tf$io$read_file(x$img)),
                mask = tf$image$decode_png(tf$io$read_file(x$mask)))
) 

# make sure tensors have the correct datatype
train_ds <- dataset_map(
  train_ds, function(x) 
    list_modify(x, 
                img = tf$image$convert_image_dtype(x$img, dtype = tf$float32),
                mask = tf$image$convert_image_dtype(x$mask, dtype = tf$float32))
) 

train_ds

# resize images in case they dont fit our input shape
train_ds <- dataset_map(
  train_ds, function(x) 
    list_modify(x,
                img = tf$image$resize(x$img, size = shape(input_shape[1], input_shape[2])),
                mask = tf$image$resize(x$mask, size = shape(input_shape[1], input_shape[2]))
    )
)

train_ds


#########################################################

## AUGMENTATION
# spectral augmentation function: alter the input image
spectral_aug <- function(img) {
  img <- tf$image$random_brightness(img, max_delta = 0.3)
  img <- tf$image$random_contrast(img, lower = 0.8, upper = 1.1)
  img <- tf$image$random_saturation(img, lower = 0.8, upper = 1.1)
  
  # value still must be within 0 and 1
  img <- tf$clip_by_value(img, 0, 1)
}

# augmentation 1: flip left right, including random change of saturation, brightness and contrast
# only modifiyng img
aug <- dataset_map(
  train_ds, function(x) 
    list_modify(x, img = spectral_aug(x$img))
)

# flipping for both img and mask
aug <- dataset_map(
  aug, function(x) 
    list_modify(x, 
                img = tf$image$flip_left_right(x$img),
                mask = tf$image$flip_left_right(x$mask))
)

# double our original dataset (original + augmentated data)
train_ds_aug <- dataset_concatenate(train_ds, aug)

# augmentation 2: flip up down, including random change of saturation, brightness and contrast
aug <- dataset_map(
  train_ds, function(x) 
    list_modify(x, img = spectral_aug(x$img))
)

aug <- dataset_map(
  aug, function(x) 
    list_modify(x, 
                img = tf$image$flip_up_down(x$img),
                mask = tf$image$flip_up_down(x$mask))
)

# triple our original dataset (original + augmentated data 1 + augmentated data 2)
train_ds_aug <- dataset_concatenate(train_ds_aug, aug)

# augmentation 3: flip left right AND up down, including random change of saturation, brightness and contrast
aug <- dataset_map(train_ds, function(x) 
  list_modify(x, img = spectral_aug(x$img))
)

aug <- dataset_map(aug, function(x) 
  list_modify(x, img = tf$image$flip_left_right(x$img),
              mask = tf$image$flip_left_right(x$mask))
)

aug <- dataset_map(aug, function(x) 
  list_modify(x, img = tf$image$flip_up_down(x$img),
              mask = tf$image$flip_up_down(x$mask))
)
# quatruple our original dataset (original + augmentated data 1 + augmentated data 2)
train_ds_aug <- dataset_concatenate(train_ds_aug, aug)



# shuffle and create batches
train_ds <- dataset_shuffle(train_ds_aug, buffer_size = batch_size*128)
train_ds <- dataset_batch(train_ds, batch_size)
train_ds <-  dataset_map(train_ds, unname) 

train_ds

# preprocess validation as above, no augmentation
val_ds <- tensor_slices_dataset(testing(data))
val_ds <- dataset_map(
  val_ds, function(x) 
    list_modify(x, 
                #img = tf$image$decode_jpeg(tf$io$read_file(x$img)),
                #mask = tf$image$decode_jpeg(tf$io$read_file(x$mask)))
                img = tf$image$decode_png(tf$io$read_file(x$img)),
                mask = tf$image$decode_png(tf$io$read_file(x$mask)))
) 

val_ds <- dataset_map(
  val_ds, function(x) 
    list_modify(x, 
                img = tf$image$convert_image_dtype(x$img, dtype = tf$float32),
                mask = tf$image$convert_image_dtype(x$mask, dtype = tf$float32))
) 

val_ds <- dataset_map(
  val_ds, function(x) 
    list_modify(x,
                img = tf$image$resize(x$img, size = shape(input_shape[1], input_shape[2])),
                mask = tf$image$resize(x$mask, size = shape(input_shape[1], input_shape[2]))
    )
)
val_ds <- dataset_batch(val_ds, batch_size)
val_ds <-  dataset_map(val_ds, unname) 

###############################################################################


## NETWORK DESIGN: UNET
l2 <- 0.03 #0.02 0.01
input_tensor <- layer_input(shape = input_shape)

# contracting path
#conv block 1
unet_tensor <- layer_conv_2d(input_tensor, filters = 64, kernel_size = c(3,3), padding = "same", activation = "relu",
                             kernel_regularizer = regularizer_l2(l2))
conc_tensor2 <- layer_conv_2d(unet_tensor, filters = 64,kernel_size = c(3,3), padding = "same", activation = "relu",
                              kernel_regularizer = regularizer_l2(l2))
unet_tensor <- layer_max_pooling_2d(conc_tensor2)

#conv block 2
unet_tensor <- layer_conv_2d(unet_tensor, filters = 128,kernel_size = c(3,3), padding = "same",activation = "relu",
                             kernel_regularizer = regularizer_l2(l2))
conc_tensor1 <- layer_conv_2d(unet_tensor, filters = 128,kernel_size = c(3,3), padding = "same",activation = "relu",
                              kernel_regularizer = regularizer_l2(l2))
unet_tensor <- layer_max_pooling_2d(conc_tensor1)

#bottom curve
unet_tensor <- layer_conv_2d(unet_tensor, filters = 256, kernel_size = c(3,3), padding = "same",activation = "relu",
                             kernel_regularizer = regularizer_l2(l2))
unet_tensor <- layer_conv_2d(unet_tensor, filters = 256, kernel_size = c(3,3), padding = "same",activation = "relu",
                             kernel_regularizer = regularizer_l2(l2))

# expanding path begins
# upsampling block 1
unet_tensor <- layer_conv_2d_transpose(unet_tensor, filters = 128, kernel_size = c(2,2), strides = 2, padding = "same",
                                       kernel_regularizer = regularizer_l2(l2))
unet_tensor <- layer_concatenate(list(conc_tensor1,unet_tensor))
unet_tensor <- layer_conv_2d(unet_tensor, filters = 128, kernel_size = c(3,3),padding = "same", activation = "relu",
                             kernel_regularizer = regularizer_l2(l2))
unet_tensor <- layer_conv_2d(unet_tensor, filters = 128, kernel_size = c(3,3),padding = "same", activation = "relu",
                             kernel_regularizer = regularizer_l2(l2))

# upsampling block 2
unet_tensor <- layer_conv_2d_transpose(unet_tensor,filters = 64,kernel_size = c(2,2),strides = 2,padding = "same",
                                       kernel_regularizer = regularizer_l2(l2))
unet_tensor <- layer_concatenate(list(conc_tensor2,unet_tensor))
unet_tensor <- layer_conv_2d(unet_tensor, filters = 64, kernel_size = c(3,3),padding = "same", activation = "relu",
                             kernel_regularizer = regularizer_l2(l2))
unet_tensor <- layer_conv_2d(unet_tensor, filters = 64, kernel_size = c(3,3),padding = "same", activation = "relu",
                             kernel_regularizer = regularizer_l2(l2))

# output
unet_tensor <- layer_conv_2d(unet_tensor,filters = 1,kernel_size = 1, activation = "sigmoid")
#unet_tensor <- layer_conv_2d(unet_tensor,filters = 3,kernel_size = 1, activation = "softmax")

unet_model <- keras_model(inputs = input_tensor, outputs = unet_tensor)
unet_model
# compile the model
compile(
  unet_model,
  optimizer = optimizer_rmsprop(learning_rate = 1e-5),
  #loss = "binary_crossentropy",
  loss = "categorical_crossentropy",
  metrics = c(metric_binary_accuracy)
  #metrics = c(metric_categorical_accuracy)
)

# train it
run <- fit(
  unet_model,
  train_ds,
  epochs = 2,
  validation_data = val_ds
)

###

#keras::save_model_hdf5(unet_model, filepath = "ice_wedges_seg_imagenet_model_nofreeze.h5")

#summary(unet_model)

# OR load:
#unet_model <- load_model_hdf5(filepath = "D:/WagnerL/ice_wedges/ice_wedges_seg_imagenet_model_nofreeze.h5")

# evlation: accuarcy and loss
evaluate(unet_model, val_ds)

# create suitable format for full scene
input_shape <- c(448,448,3)
ds_data <- rbind(
  data.frame(
    img  = list.files("C:/Users/Cornelia/Documents/Studium/EAGLE/2Semester/AdvancedProgramming/Project/sat_tiles_0_1_29_9_png_3bands", pattern=".png$", full.names = T),
    mask = list.files("C:/Users/Cornelia/Documents/Studium/EAGLE/2Semester/AdvancedProgramming/Project/mask_tiles_0_1_29_9_png", pattern=".png$", full.names = T)
  )
)
ds_data

ds <- tensor_slices_dataset(ds_data)
ds <- dataset_map(
  ds, function(x) 
    list_modify(x, 
                img = tf$image$decode_png(tf$io$read_file(x$img)),
                mask = tf$image$decode_png(tf$io$read_file(x$mask)))
) 

ds <- dataset_map(
  ds, function(x) 
    list_modify(x, 
                img = tf$image$convert_image_dtype(x$img, dtype = tf$float32),
                mask = tf$image$convert_image_dtype(x$mask, dtype = tf$float32))
) 

ds <- dataset_map(
  ds, function(x) 
    list_modify(x,
                img = tf$image$resize(x$img, size = shape(input_shape[1], input_shape[2])),
                mask = tf$image$resize(x$mask, size = shape(input_shape[1], input_shape[2]))
    )
)
ds <- dataset_batch(ds, batch_size)
ds <-  dataset_map(ds, unname) 

predictions <- predict(unet_model, ds)
class(predictions)

library(jpeg)
library(png)
for (i in seq(1,20)){
  print(i)
  # test <- predictions[i,,, 1]%>% `>`(0.2)%>% k_cast("int32")
  test <- predictions[i,,, 1]%>%`>`(0.49)%>% k_cast("int32")
  #test <- predictions[i,,, 1]
  print(as.matrix(test))
  writePNG(as.matrix(test), target=paste("C:/Users/Cornelia/Documents/Studium/EAGLE/2Semester/AdvancedProgramming/Project/predictions_0_1_29_9_png/prediction_tile_", i+100, ".png", sep=""))
}





