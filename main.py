import keras

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# normalize images to 0..1
x_train = x_train.astype("float64") / 255
x_test  = x_test.astype("float64") / 255

# one hot encoding
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# add 3rd dimension because Conv2D Layer expects 3D Vektor
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# Sequentially add following layers
model = keras.models.Sequential()
# Input shape has to be given to first layer -> image size
model.add(keras.layers.Conv2D(filters=8, kernel_size=(3,3), activation="relu", input_shape=(28, 28, 1)))
model.add(keras.layers.AveragePooling2D())
model.add(keras.layers.Conv2D(filters=16, kernel_size=(3,3), activation="relu"))
model.add(keras.layers.AveragePooling2D())
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(units=10, activation="softmax"))

model.summary()

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.SGD(), metrics=["accuracy"])

model.fit(x=x_train, y=y_train, batch_size=32, epochs=1, validation_data=(x_test, y_test), verbose=2)

# model.save('model.ka')
