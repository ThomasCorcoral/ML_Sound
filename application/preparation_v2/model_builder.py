import tensorflow as tf


def builder(size):
    """The neural network responsible for the data analysis"""
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(50, 43)),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(512, activation='relu'),
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