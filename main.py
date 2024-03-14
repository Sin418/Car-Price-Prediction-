import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt  # Import matplotlib for plotting
from tensorflow.keras.layers import Normalization, Dense, InputLayer, Dropout
from tensorflow.keras.losses import MeanAbsoluteError, MeanSquaredError, Huber
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import relu, sigmoid




# Assuming df is your DataFrame
df = pd.read_csv('train.csv')
df.drop(columns=['on road old', 'on road now', 'v.id'], inplace=True)

'''print(df.head())

# Create pairplot
sns.pairplot(df[['years', 'km', 'rating', 'condition', 'economy', 'top speed', 'hp', 'torque', 'current price']], diag_kind='kde')

plt.show()
'''

tensor_data = tf.constant(df)
tensor_labels = tf.random.shuffle(tensor_data)

x = tensor_data[:, :-1]

y = tensor_data[:, -1]

y = tf.expand_dims(y, axis=1)


TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
DATASET_SIZE = len(x)


x_train = x[:int(DATASET_SIZE*TRAIN_RATIO)]
y_train = y[:int(DATASET_SIZE*TRAIN_RATIO)]
x_val = x[int(DATASET_SIZE*TRAIN_RATIO):int(DATASET_SIZE*(TRAIN_RATIO+VAL_RATIO))]
y_val = y[int(DATASET_SIZE*TRAIN_RATIO):int(DATASET_SIZE*(TRAIN_RATIO+VAL_RATIO))]
x_test = x[int(DATASET_SIZE*(TRAIN_RATIO+VAL_RATIO)):]
y_test = y[int(DATASET_SIZE*(TRAIN_RATIO+VAL_RATIO)):]

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=8,reshuffle_each_iteration=True).batch(32).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset = train_dataset.shuffle(buffer_size=8,reshuffle_each_iteration=True).batch(32).prefetch(tf.data.AUTOTUNE)


test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset = train_dataset.shuffle(buffer_size=8,reshuffle_each_iteration=True).batch(32).prefetch(tf.data.AUTOTUNE)

normalizer = Normalization()
normalizer.adapt(x_train)

model = tf.keras.models.Sequential([tf.keras.layers.InputLayer(input_shape=(8,)),
                                    normalizer,
                                    Dense(250, activation=relu),
                                    Dense(250, activation=relu),
                                    Dense(250, activation=relu),
                                    Dense(1)
])

model.compile(optimizer=Adam(learning_rate=1),
              loss=MeanAbsoluteError(),
              metrics=RootMeanSquaredError())


history = model.fit(train_dataset, validation_data=val_dataset, epochs = 100, verbose = 1)

plt.plot(history.history['root_mean_squared_error'])
plt.plot(history.history['val_root_mean_squared_error'])


plt.title('model performance')
plt.ylabel('rmse')
plt.xlabel('epoch')
plt.legend(['train',"val"])

plt.show()



model.evaluate(test_dataset)