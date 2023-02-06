
def create_dataset(file_names, ages, input_size, batch_size, shuffle, cache_file=None):

    # Create a Dataset object
    dataset = tf.data.Dataset.from_tensor_slices((file_names, ages))

    # Map the load_image function
    py_func = lambda file_name, ages: (tf.numpy_function(load_image, [file_name, input_size], tf.float32))
    dataset = dataset.map(py_func, num_parallel_calls=os.cpu_count())

    # # Map the normalize_img function
    dataset = dataset.map(normalize_img, num_parallel_calls=os.cpu_count())

    # # Map the preprocess_data function
    # py_func = lambda file_name, label: (tf.numpy_function(preprocess_data, [file_name, data_dir],
    #                                                          tf.float32), label)
    # dataset = dataset.map(py_func, num_parallel_calls=os.cpu_count())


    # # # Duplicate data for the autoencoder (input = output)
    # # py_funct = lambda img: (img, img)
    # # dataset = dataset.map(py_funct)

    # Cache dataset
    if cache_file:
        dataset = dataset.cache(cache_file)

    # Shuffle
    if shuffle:
        dataset = dataset.shuffle(len(file_names))

    # Repeat the dataset indefinitely
    dataset = dataset.repeat()

    # Batch
    dataset = dataset.batch(batch_size=batch_size)

    # Prefetch
    dataset = dataset.prefetch(buffer_size=1)

    return dataset

batch_size = 32
train_dataset = create_dataset(file_names = X_train, 
                    ages = age_train, 
                    input_size = IMG_SHAPE,
                    batch_size = batch_size, 
                    shuffle = False )  

validation_dataset = create_dataset(file_names = X_validation, 
                    ages = age_validation, 
                    input_size = IMG_SHAPE,
                    batch_size = batch_size, 
                    shuffle = False ) 

test_dataset = create_dataset(file_names = X_test, 
                    ages = age_test, 
                    input_size = IMG_SHAPE,
                    batch_size = batch_size, 
                    shuffle = False )    

train_steps = int(np.ceil(len(X_train) / batch_size))
validation_steps = int(np.ceil(len(X_validation) / batch_size))
test_steps = int(np.ceil(len(X_test) / batch_size))