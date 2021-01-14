import tensorflow as tf
from numpy import load


def my_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, kernel_initializer='lecun_normal', activation='selu'),
        tf.keras.layers.Dense(128, kernel_initializer='lecun_normal', activation='elu'),
        tf.keras.layers.Dense(64, kernel_initializer='lecun_normal', activation='elu'),
        tf.keras.layers.Dense(32, kernel_initializer='lecun_normal', activation='selu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model

def run_model(epoch=10):
    train_audio = load('../local_npy_files/train_audio.npy')
    train_labels = load('../local_npy_files/train_labels.npy')
    test_audio = load('../local_npy_files/test_audio.npy')
    test_labels = load('../local_npy_files/test_labels.npy')

    print("test audio size : " + str(len(test_audio)))
    print("test labels size : " + str(len(test_labels)))
    print("train audi size : " + str(len(train_audio)))
    print("train labels size : " + str(len(train_labels)))

    model = my_model()

    model.fit(train_audio, train_labels, epochs=epoch, verbose=0)
    score = model.evaluate(test_audio, test_labels, verbose=1)
    accuracy = 100 * score[1]

    return accuracy, model