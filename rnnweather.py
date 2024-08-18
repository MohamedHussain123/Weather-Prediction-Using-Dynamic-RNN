import tensorflow as tf
import numpy as np
import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm

# Data loading and preprocessing
filename = 'D:/ML/weather.npz'  # Specify your filename
data = np.load(filename)
daily = data['daily']
weekly = data['weekly']
num_weeks = len(weekly)
dates = np.array([datetime.datetime.strptime(str(int(d)), '%Y%m%d') for d in weekly[:, 0]])

def assign_season(date):
    month = date.month
    if 3 <= month < 6:
        season = 0
    elif 6 <= month < 9:
        season = 1
    elif 9 <= month < 12:
        season = 2
    elif month == 12 or month < 3:
        season = 3
    return season

num_classes = 4
num_inputs = 5
labels = np.zeros([num_weeks, num_classes])
for i, d in enumerate(dates):
    labels[i, assign_season(d)] = 1

train = weekly[:, 1:]
train = train - np.average(train, axis=0)
train = train / train.std(axis=0)

# Reshape data for RNN [samples, time_steps, features]
X = train.reshape((num_weeks, 1, num_inputs))  # Reshape to (num_weeks, 1, num_inputs)
y = labels.reshape((num_weeks, num_classes))   # Reshape to (num_weeks, num_classes)

# Split data into training and testing
split_index = int(num_weeks * 0.8)  # 80% for training, 20% for testing
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

def create_dynamic_rnn_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.RNN(tf.keras.layers.SimpleRNNCell(50), return_sequences=False),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

@tf.function
def train_step(model, X_batch, y_batch):
    with tf.GradientTape() as tape:
        y_pred = model(X_batch, training=True)
        loss = model.compute_loss(X_batch, y_batch, y_pred)
    grads = tape.gradient(loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
    
    # Update metrics
    for metric in model.metrics:
        metric.update_state(y_batch, y_pred)
    
    return loss

def train_model(model, X_train, y_train, X_test, y_test, epochs):
    steps_per_epoch = int(np.ceil(len(X_train) / 1))  # batch_size = 1

    # Lists to store accuracy values
    train_accuracies = []
    test_accuracies = []

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        
        # Create a tqdm progress bar for batches
        batch_progress = tqdm(range(steps_per_epoch), desc='Training', position=0, leave=True)
        
        for step in batch_progress:
            # Fit the model on one batch
            batch_start = step * 1
            batch_end = min(batch_start + 1, len(X_train))
            
            X_batch = X_train[batch_start:batch_end, :, :]
            y_batch = y_train[batch_start:batch_end, :]
            
            # Train step
            loss = train_step(model, X_batch, y_batch)
            
            # Update progress bar with loss and metrics
            # Calculate metrics manually
            train_metrics = model.evaluate(X_train, y_train, verbose=0)
            test_metrics = model.evaluate(X_test, y_test, verbose=0)
            metrics = {'loss': train_metrics[0], 'accuracy': train_metrics[1]}
            batch_progress.set_postfix({'loss': loss.numpy(), **metrics})
        
        # Record accuracies
        train_accuracy = model.evaluate(X_train, y_train, verbose=0)[1]
        test_accuracy = model.evaluate(X_test, y_test, verbose=0)[1]
        
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        
        print(f"Training Accuracy: {train_accuracy:.4f}, Testing Accuracy: {test_accuracy:.4f}")

    return train_accuracies, test_accuracies

# Train and evaluate Dynamic RNN model
dynamic_rnn_model = create_dynamic_rnn_model(input_shape=(1, num_inputs), num_classes=num_classes)
print("Training Dynamic RNN Model")
train_accuracies, test_accuracies = train_model(dynamic_rnn_model, X_train, y_train, X_test, y_test, epochs=10)

# Plot training and testing accuracy
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Training Accuracy')
plt.plot(range(1, len(test_accuracies) + 1), test_accuracies, label='Testing Accuracy')
plt.title('Training and Testing Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Predict
pred = np.argmax(dynamic_rnn_model.predict(X_test), axis=1)
conf = np.zeros([num_classes, num_classes])

for p, t in zip(pred, np.argmax(y_test, axis=1)):
    conf[t, p] += 1

# Plot confusion matrix for Dynamic RNN
plt.figure(figsize=(10, 7))
plt.imshow(conf, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix - Dynamic RNN')
plt.colorbar()
classes = [f'Season {i}' for i in range(num_classes)]
plt.xticks(np.arange(num_classes), classes, rotation=45)
plt.yticks(np.arange(num_classes), classes)

for i in range(num_classes):
    for j in range(num_classes):
        plt.text(j, i, int(conf[i, j]), 
                 ha='center', va='center', color='black')

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
