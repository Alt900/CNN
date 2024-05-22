import os
import tensorflow as tf
import PIL.Image as I
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

class CNN():
    def __init__(self,datasetpath,batch_size,epochs,image_size):
        self.datasetpath=datasetpath
        self.batch_size=batch_size
        self.epochs=epochs
        self.image_size=image_size

    def __run_setup(self):
        self.training_dataset=keras.preprocessing.image_dataset_from_directory(
            directory=f"{os.getcwd()}\\{self.datasetpath}",
            labels='inferred',
            image_size=self.image_size,
            batch_size=32,
            seed=123,
            validation_split=0.2,
            subset="training"
        )
        self.testing_dataset=keras.preprocessing.image_dataset_from_directory(
            directory=f"{os.getcwd()}\\{self.datasetpath}",
            labels='inferred',
            image_size=self.image_size,
            batch_size=32,
            seed=123,
            validation_split=0.2,
            subset="validation"
        )

    def resize_dataset(self,dimensiontuple):
        cwd=os.getcwd()
        for y in [x[0] for x in os.walk(f"{os.getcwd()}\\{self.datasetpath}")][1:]:
            for x in os.listdir(y):
                os.chdir(y)
                x=os.fsdecode(x)
                print(x)
                if x.endswith(".py"):
                    continue
                i=I.open(x)
                i=i.resize(dimensiontuple)
                i.save(x)
        os.chdir(cwd)
    
    def train(self):
        self.__run_setup()
        self.model=models.Sequential([
            layers.Conv2D(32, (3, 3), activation='tanh', input_shape=(self.image_size[0], self.image_size[1], 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='leaky_relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='leaky_relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation='sigmoid'),
            layers.Dense(len(
                [x[0] for x in os.walk(f"{os.getcwd()}\\{self.datasetpath}")][1:]#the number of sub-directory classes in data dir
            ))
        ])

        self.model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

        history = self.model.fit(
            self.training_dataset,
            validation_data=self.testing_dataset,
            epochs=self.epochs
        )

        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs_range = range(self.epochs)

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()

    def predict(self,image):
        image=tf.expand_dims(tf.keras.utils.img_to_array(image),0)
        score=tf.nn.softmax(self.model.predict(image))[0]
        print(self.training_dataset.class_names)
        print(np.argmax(score))
        print(score)
        return f"This images belongs to the class {self.training_dataset.class_names[np.argmax(score)]}\nsureness percentage of %{100*np.max(score)}"


image_size_tuple=(225,225)

C=CNN("data",5,100,image_size_tuple)

#C.resize_dataset(image_size_tuple) #changes the images to a 65x65 image this makes the dataset images uniform sizes 

C.train()

print(C.predict(
    tf.keras.utils.load_img(
        "testimage.extension",
        target_size=image_size_tuple
    )
))