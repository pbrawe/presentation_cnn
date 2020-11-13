from tensorflow import keras
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

model = keras.models.load_model('model.ka', compile=True)

test = x_test[17]

test = test.reshape(1, 28,28,1)

plt.imshow(x_test[17].reshape(28, 28))
plt.show()
pr = model.predict_classes(test)

print(pr)