# Inicio ============================================================
  library(keras)   
  library(magick)


# Cargo mi modelo ============================================================
  my_model = load_model_hdf5("./model_checkpoints.h5" , compile=TRUE)
  load("./fruits_classes_indices.RData")
  df_classes_indices <- data.frame(indices = unlist(classes_indices), prob=rep(0,length(classes_indices)) )
  
  # classes_indices_df <- data.frame(indices = unlist(classes_indices))
  # classes_indices_df <- classes_indices_df[order(classes_indices_df$indices), , drop = FALSE]
  





  
  
# Prueba de a una sola imagen ===============================
  # img_url <- 'https://upload.wikimedia.org/wikipedia/commons/thumb/8/8a/Banana-Single.jpg/272px-Banana-Single.jpg'
  # img_url <- 'https://upload.wikimedia.org/wikipedia/commons/5/53/Strawberry_gariguette_DSC03063.JPG'
  # img_url <- 'https://cdn.pixabay.com/photo/2010/12/13/09/51/clementine-1792_1280.jpg'
  img_url <- 'https://www.zonefresh.com.au/wp-content/uploads/52.jpg' #papaya
  
  img_to_disc <- image_read(img_url)
  plot(img_to_disc)
  image_write(img_to_disc,'./imagen_prueba.jpg')
  
  
  
  img <- image_load("./imagen_prueba.jpg", target_size = c(100,100))
  img <- image_to_array(img)
  img <- img/255
  plot(as.raster(img))
  img <- array_reshape(img, c(1, dim(img)))

  
  df_classes_indices$prob <- as.vector(predict_proba(my_model,img, steps=1))
  clase_pred <- df_classes_indices[which.max(df_classes_indices$prob),1]
  paste0(rownames(df_classes_indices)[clase_pred+1],' prob :', round(df_classes_indices[clase_pred+1,2],4) )
  # predict_classes(my_model,img, steps=1)
  plot(df_classes_indices$indice,df_classes_indices$prob, type = 'b', main=paste0(rownames(df_classes_indices)[clase_pred+1],':', round(df_classes_indices[clase_pred+1,2],4) ))
  
  
  
# Pruebas a partir de las imagenes guardadas ===============================
#   test_datagen <- image_data_generator(rescale = 1/255)
#   
#   test_generator <- flow_images_from_directory(
#     test_image_files_path,
#     test_datagen,
#     target_size = c(100,100),
#     class_mode = 'categorical')
#   
#   predictions <- as.data.frame(predict_generator(my_model, test_generator, steps = 1))
#   
#   colnames(predictions) <- rownames(classes_indices_df)
#   
#   t(round(predictions, digits = 2)) 
# 
#   for (i in 1:nrow(prediction)) {
#     cat(i, ":")
#     print(unlist(which.max(prediction[i, ])))
#   }