import tensorflow as tf
import convert_data as cd

num_rows = 10
num_columns = 4
num_channels = 1

train_audio, train_labels = cd.conv_data()
# print(y_train[10])
# x_train = x_train.reshape(x_train.shape[0], num_rows, num_columns, num_channels)
# x_test = x_test.reshape(x_test.shape[0], num_rows, num_columns, num_channels)

# num_labels = yy.shape[1]
filter_size = 2
#
model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128),
    tf.keras.layers.Dense(128),
    tf.keras.layers.Dense(64),
    tf.keras.layers.Dense(32)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

model.fit(train_audio, train_labels, epochs=1000)

score = model.evaluate(train_audio, train_labels, verbose=1)
accuracy = 100*score[1]

print("Pre-training accuracy: %.4f%%" % accuracy)
