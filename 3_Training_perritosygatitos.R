# Inicio ============================================================
library(keras)
 

# list of classes to model
# classes_list <- c("Kiwi", "Banana", "Apricot", "Avocado", "Cocos", "Clementine", "Mandarine", "Orange",
                # "Limes", "Lemon", "Peach", "Plum", "Raspberry", "Strawberry", "Pineapple", "Pomegranate")

classes_list <- c("perrito", "gatito")


# number of output classes
output_n <- length(classes_list)

# image size to scale down to (original images are 100 x 100 px)
img_width <- 40
img_height <- 40
target_size <- c(img_width, img_height)

# RGB = 3 channels
channels <- 3

# path to image folders
  # frutas
  # train_image_files_path <- "/Users/eugenio/Datasets/image_recognition/fruits-360/Training/"
  # valid_image_files_path <- "/Users/eugenio/Datasets/image_recognition/fruits-360/Test/"
  
  # perritos y gatitos
  train_image_files_path <- "/Users/eugenio/Datasets/image_recognition/perritos_y_gatitos/train/"
  valid_image_files_path <- "/Users/eugenio/Datasets/image_recognition/perritos_y_gatitos/validation/"



# Loading images ============================================================

  # Generadores de Imagenes ==================================================
  # generadores de lotes de imagenes con data-augmentation en tiempo real.
  # definimos uno para training y otro para test.

  # optional data augmentation
  train_data_gen = image_data_generator(
    rescale = 1/255 #,
    #rotation_range = 40,
    #width_shift_range = 0.2,
    #height_shift_range = 0.2,
    #shear_range = 0.2,
    #zoom_range = 0.2,
    #horizontal_flip = TRUE,
    #fill_mode = "nearest"
  )
  
  # Validation data shouldn't be augmented! But it should also be scaled.
  valid_data_gen <- image_data_generator(
    rescale = 1/255
  )  


  # Carga de imagenes  ==================================================
  # training images
  train_image_array_gen <- flow_images_from_directory(train_image_files_path, 
                                                      train_data_gen,
                                                      target_size = target_size,
                                                      class_mode = "categorical",
                                                      classes = classes_list,
                                                      seed = 42)
  
  # validation images
  valid_image_array_gen <- flow_images_from_directory(valid_image_files_path, 
                                                      valid_data_gen,
                                                      target_size = target_size,
                                                      class_mode = "categorical",
                                                      classes = classes_list,
                                                      seed = 42)
  
  
  
  ## Number of images per class:
  table(factor(train_image_array_gen$classes))

  
  ## Class label vs index mapping:
  train_image_array_gen$class_indices

  fruits_classes_indices <- train_image_array_gen$class_indices
  save(fruits_classes_indices, file = "./fruits_classes_indices.RData")
  
  
# Define Model ============================================================
  
  # number of training samples
  train_samples <- train_image_array_gen$n
  # number of validation samples
  valid_samples <- valid_image_array_gen$n
  
  # define batch size and number of epochs
  batch_size <- 32
  epochs <- 5
  
  
  # initialise model
  model <- keras_model_sequential()
  
  # add layers
  model %>%
    layer_conv_2d(filter = 32, kernel_size = c(3,3), padding = "same", input_shape = c(img_width, img_height, channels)) %>%
    layer_activation("relu") %>%
    
    # Second hidden layer
    layer_conv_2d(filter = 16, kernel_size = c(3,3), padding = "same") %>%
    layer_activation_leaky_relu(0.5) %>%
    layer_batch_normalization() %>%
    
    # Use max pooling
    layer_max_pooling_2d(pool_size = c(2,2)) %>%
    layer_dropout(0.25) %>%
    
    # Flatten max filtered output into feature vector 
    # and feed into dense layer
    layer_flatten() %>%
    layer_dense(100) %>%
    layer_activation("relu") %>%
    layer_dropout(0.5) %>%
    
    # Outputs from dense layer are projected onto output layer
    layer_dense(output_n) %>% 
    layer_activation("softmax")
  
  # compile
  model %>% compile(
    loss = "categorical_crossentropy",
    optimizer = optimizer_rmsprop(lr = 0.0001, decay = 1e-6),
    metrics = "accuracy"
  )
  
  
# Fitting the model ============================================================
  # fit
  hist <- model %>% fit_generator(
    # training data
    train_image_array_gen,
    
    # epochs
    steps_per_epoch = as.integer(train_samples / batch_size), 
    epochs = epochs, 
    
    # validation data
    validation_data = valid_image_array_gen,
    validation_steps = as.integer(valid_samples / batch_size),
    
    # print progress
    verbose = 2,
    callbacks = list(
      # save best model after every epoch
      callback_model_checkpoint("./models_checkpoints.h5", save_best_only = TRUE),
      # only needed for visualising with TensorBoard
      callback_tensorboard(log_dir = "./logs")
    )
  )
  
  
# Results ============================================================
  
  plot(hist)
  tensorboard("./logs")
  sessionInfo()
  