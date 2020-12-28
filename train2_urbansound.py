# Load various imports
from numpy import load
import tensorflow as tf


def first_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='selu'),
        tf.keras.layers.Dense(128, activation='selu'),
        tf.keras.layers.Dense(128, activation='selu'),
        tf.keras.layers.Dense(64, activation='selu'),
        tf.keras.layers.Dense(32, activation='selu')
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model

def second_model():
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

def third_model():
    initializer = tf.keras.initializers.Orthogonal()
    initializer2 = tf.keras.initializers.Identity()
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(256, activation='elu'),
        tf.keras.layers.Dense(128, kernel_initializer=initializer, activation='selu'),
        tf.keras.layers.Dense(128, kernel_initializer=initializer, activation='elu'),
        tf.keras.layers.Dense(64, kernel_initializer=initializer, activation='elu'),
        tf.keras.layers.Dense(32, kernel_initializer=initializer, activation='selu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model


if __name__ == "__main__":
    corresp_labels = ["air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling", "engine_idling",
                      "gun_shot", "jackhammer", "siren", "street_music"]

    train_audio = load('./arrays/train_audio.npy')
    train_labels = load('./arrays/train_labels.npy')
    test_audio = load('./arrays/test_audio.npy')
    test_labels = load('./arrays/test_labels.npy')

    first = 0
    second = 0
    third = 0

    accuracy1 = 0

    for i in range(10):
        # model = first_model()
        #
        # model.fit(train_audio, train_labels, epochs=10, verbose=0)
        # score = model.evaluate(test_audio, test_labels, verbose=0)
        # accuracy1 = 100 * score[1]
        # print("Pre-training accuracy: %.4f%%" % accuracy1)

        model = second_model()

        model.fit(train_audio, train_labels, epochs=10, verbose=0)
        score = model.evaluate(test_audio, test_labels, verbose=0)
        accuracy2 = 100 * score[1]
        # print("Pre-training accuracy: %.4f%%" % accuracy2)

        model = third_model()

        model.fit(train_audio, train_labels, epochs=10, verbose=0)
        score = model.evaluate(test_audio, test_labels, verbose=0)
        accuracy3 = 100 * score[1]
        # print("Pre-training accuracy: %.4f%%" % accuracy3)
        first = first + accuracy1
        second = second + accuracy2
        third = third + accuracy3

        if (accuracy1 > accuracy2) & (accuracy1 > accuracy3):
            print("Le premier gagne")
        elif (accuracy2 > accuracy3) & (accuracy2 > accuracy1):
            print("Le second gagne")
        else:
            print("Le troisième gagne")
    print("Score total du premier model : ", str(first))
    print("Score total du second model : ", str(second))
    print("Score total du troisième model : ", str(third))

