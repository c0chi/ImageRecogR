# Inicio ============================================================
  library(keras)   
  library(magick)

  my_model = model
  my_model = load_model_hdf5("./fruits_checkpoints.h5" , compile=FALSE)
  



# Guardo imagenes de internet en el disco ===============================
img_url <- 'https://cdn.pixabay.com/photo/2010/12/13/09/51/clementine-1792_1280.jpg'
# img_url <- 'https://upload.wikimedia.org/wikipedia/commons/thumb/8/8a/Banana-Single.jpg/272px-Banana-Single.jpg'
img_to_disc <- image_read(img_url)
img_path <- file.path('./imagen_prueba.jpg')
image_write(img_to_disc, img_path)
#plot(as.raster(img))


# las levanta y predice la clase ===============================

img <- image_load("./imagen_prueba.jpg", target_size = c(40,40))
img <- image_to_array(img)
img <- array_reshape(img, c(1, dim(img)))
img <- imagenet_preprocess_input(img)


img

predict_proba(my_model, img)
pred


