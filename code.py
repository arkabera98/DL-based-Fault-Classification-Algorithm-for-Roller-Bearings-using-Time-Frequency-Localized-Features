# Importing the Keras libraries and packages

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint


# CNN architecture

classifier = Sequential()

classifier.add(Conv2D(32, (3, 3), input_shape = (256, 256, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2))))

classifier.add(Conv2D(128, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Conv2D(128, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Flatten())

classifier.add(Dense(units = 256, activation = 'relu'))
classifier.add(Dropout(rate = 0.20))
classifier.add(Dense(units = 5, activation = 'softmax'))

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


# Fitting images to CNN

image_generator = ImageDataGenerator(validation_split = 0.3)    

batch_size = 32

training_set = image_generator.flow_from_directory(batch_size = batch_size,
                                                   directory = '/content/drive/My Drive/DATA/Training',
                                                   shuffle = True,
                                                   target_size = (256, 256),
                                                   subset = "training",
                                                   class_mode = 'categorical')

test_set = image_generator.flow_from_directory(batch_size = batch_size,
                                               directory = '/content/drive/My Drive/DATA/Training',
                                               shuffle = True,
                                               target_size = (256, 256), 
                                               subset = "validation",
                                               class_mode = 'categorical')

reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5,
                             patience = 5, min_lr = 0.000001)
                             
es = EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 5, restore_best_weights = True)

history = classifier.fit(training_set,
                         steps_per_epoch = 21000//batch_size,
                         epochs = 20,
                         validation_data = test_set,
                         validation_steps = 9000//batch_size,
                         callbacks = [reduce_lr,es])
