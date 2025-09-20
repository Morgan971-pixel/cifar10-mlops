# This is a test comment to trigger a new CI/CD run.
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras_tuner.tuners import RandomSearch
from keras_tuner import HyperParameters
from tensorflow.keras.callbacks import EarlyStopping

# 1. Load and preprocess the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values to be between 0 and 1
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Convert labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

# 2. Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
)
datagen.fit(x_train)

# 3. Model Building with Transfer Learning
def build_model(hp):
    # Load the VGG16 model, pre-trained on ImageNet, without the top classification layers
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

    # Freeze the layers of the base model
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom layers on top of the base model
    x = Flatten()(base_model.output)
    x = Dense(hp.Int('units', min_value=256, max_value=1024, step=256), activation='relu')(x)
    x = Dropout(hp.Float('dropout', min_value=0.2, max_value=0.5, step=0.1))(x)
    predictions = Dense(10, activation='softmax')(x)

    # Create the final model
    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile the model
    model.compile(
        optimizer=Adam(hp.Choice('learning_rate', values=[1e-3, 1e-4])),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# 4. Hyperparameter Tuning using Keras Tuner
tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,
    executions_per_trial=3,
    directory='keras_tuner_dir',
    project_name='cifar10_cnn'
)

tuner.search_space_summary()

# We use a smaller subset of the data for hyperparameter tuning to speed up the process
x_train_small = x_train[:10000]
y_train_small = y_train[:10000]
x_val_small = x_train[10000:12000]
y_val_small = y_train[10000:12000]

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

tuner.search(
    datagen.flow(x_train_small, y_train_small, batch_size=64),
    epochs=10,
    validation_data=(x_val_small, y_val_small),
    callbacks=[early_stopping]
)

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete. The optimal number of units in the first dense layer is {best_hps.get('units')} and the optimal learning rate for the optimizer is {best_hps.get('learning_rate')}.
""")

# 5. Model Training and Evaluation
# Build the model with the optimal hyperparameters and train it on the entire dataset
model = tuner.hypermodel.build(best_hps)
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=64),
    epochs=50,
    validation_data=(x_test, y_test),
    callbacks=[early_stopping]
)

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test accuracy: {accuracy}')

# Save the best model
model.save('cifar10_cnn_model.h5')

# --- Azure ML Model Registration ---
# This part will only run when executed in an Azure ML environment
# It assumes that the necessary environment variables are set by Azure ML
try:
    from azure.ai.ml import MLClient
    from azure.ai.ml.entities import Model
    from azure.identity import DefaultAzureCredential
    import os

    # Connect to Azure ML (using environment variables set by Azure ML)
    ml_client = MLClient(
        DefaultAzureCredential(),
        subscription_id=os.environ.get("AZUREML_ARM_SUBSCRIPTION"),
        resource_group_name=os.environ.get("AZUREML_ARM_RESOURCEGROUP"),
        workspace_name=os.environ.get("AZUREML_ARM_WORKSPACE_NAME"),
    )

    # Register the model
    model_name = "cifar10-cnn-model"
    model_path = "cifar10_cnn_model.h5"

    registered_model = ml_client.models.create_or_update(
        Model(name=model_name, path=model_path, description="CIFAR-10 CNN with Transfer Learning"),
    )
    print(f"Model {registered_model.name} registered with version {registered_model.version}")

except Exception as e:
    print(f"Could not register model with Azure ML: {e}")