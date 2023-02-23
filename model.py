import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random

imgsize = (100,140)
dataset, validset = tf.keras.utils.image_dataset_from_directory(
    'AHBase\AHDBase_TrainingSet',
    label_mode='int', image_size=imgsize, color_mode='grayscale',validation_split=0.1666666666,subset='both',
    seed=5
)

# display an image for visualization
def tensor_to_image(tensor):
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)

# this bit to see first image for debugging  
# for x, y in dataset:
#   print(type(x[0]), x[0].dtype)
#   img = tensor_to_image(x[0][:,:,0])
#   img.save('mf.png')
#   break


model = tf.keras.models.Sequential(
    [   tf.keras.layers.Rescaling(1./255),
        tf.keras.layers.Flatten(input_shape=(100,140)),
        tf.keras.Input(shape=(14000,)),
        tf.keras.layers.Dense(50, activation='linear'),
        tf.keras.layers.Dense(25, activation='linear'),
        tf.keras.layers.Dense(10)
     ]
)
initial_learning_rate = 0.0001
learningRate = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=100000,
    decay_rate=0.7)

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer= tf.keras.optimizers.Adam(learning_rate=learningRate),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
)

history = model.fit(
    dataset,
    validation_data=validset,
    epochs=30
)

plt.plot(history.history['loss'][1::])
plt.plot(history.history['val_loss'][1::])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.figure()
plt.plot(history.history['sparse_categorical_accuracy'][1::])
plt.plot(history.history['sparse_categorical_accuracy'][1::])
plt.title('model accuracy')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['acc', 'val_acc'], loc='upper left')
plt.show()

testset = tf.keras.utils.image_dataset_from_directory(
    'AHBase\AHDBase_TestingSet',
    label_mode='int', image_size=imgsize, color_mode='grayscale',
    shuffle=False
)
testset = testset.shuffle(1000, reshuffle_each_iteration=False)
thing = model.evaluate(testset)
print(thing)
Y = np.concatenate([y for x, y in testset], axis=0)
X =  np.concatenate([x for x, y in testset], axis=0)
Yhat = model.predict(testset)
print(X.shape, Y.shape, Yhat.shape)
C = []
for i in range(len(Y)):
    ysoft = tf.nn.softmax(Yhat[i])
    ymax = np.argmax(ysoft)
    C.append(ymax == Y[i])

mean =np.sum(C)/len(C)
print(f'correct={mean*100}%') 

fig, ax = plt.subplots(4,8)
for i in range(4):
    for j in range(8):
        index = random.randint(0,9999)
        img = tensor_to_image(X[index][:,:,0])
        ax[i,j].imshow(img)
        ysoft = tf.nn.softmax(Yhat[index])
        ymax = np.argmax(ysoft)
        ax[i,j].set_title(f'Y:{Y[index]},P:{ymax}')
        ax[i,j].axis('off')
plt.show()

model.save('mymodel/mymodel')