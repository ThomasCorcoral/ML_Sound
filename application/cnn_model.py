import tensorflow as tf
from numpy import load


def my_model(size):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, kernel_initializer='lecun_normal', activation='selu'),
        tf.keras.layers.Dense(128, kernel_initializer='lecun_normal', activation='elu'),
        tf.keras.layers.Dense(64, kernel_initializer='lecun_normal', activation='elu'),
        tf.keras.layers.Dense(32, kernel_initializer='lecun_normal', activation='selu'),
        tf.keras.layers.Dense(size, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model


def read_labels():
    class_label = []
    with open('../local_saves/class_label.txt', 'r') as filehandle:
        for line in filehandle:
            current_place = line[:-1]
            class_label.append(current_place)
    return class_label


def run_model(epoch=10):
    train_audio = load('../local_saves/train_audio.npy')
    train_labels = load('../local_saves/train_labels.npy')
    test_audio = load('../local_saves/test_audio.npy')
    test_labels = load('../local_saves/test_labels.npy')
    print("test audio size : " + str(len(test_audio)))
    print("test labels size : " + str(len(test_labels)))
    print("train audi size : " + str(len(train_audio)))
    print("train labels size : " + str(len(train_labels)))

    class_label = read_labels()
    class_label = list(dict.fromkeys(class_label))

    model = my_model(len(class_label))
    model.fit(train_audio, train_labels, epochs=epoch, verbose=0)
    score = model.evaluate(test_audio, test_labels, verbose=1)
    accuracy = 100 * score[1]
    json_file = model.to_json()
    with open("../local_saves/model.json", "w") as file:
        file.write(json_file)
    with open("../local_saves/accuracy.txt", "w") as file:
        file.write(str(accuracy))
    model.save_weights("../local_saves/model.h5")

    return accuracy, model