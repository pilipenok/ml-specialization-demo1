from __future__ import division
from __future__ import print_function

from absl import logging
import tensorflow as tf
import tensorflow_transform as tft

from models.keras.baseline_advanced import features
from models.keras.baseline_advanced.features import transformed_name as t
from models.keras.baseline_advanced import constants
from tfx_bsl.tfxio import dataset_options


def _get_serve_tf_examples_fn(model, tf_transform_output):
    """Returns a function that parses a serialized tf.Example and applies TFT."""

    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        """Returns the output to be used in the serving signature."""
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop('relative_demand')
        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)

        transformed_features = model.tft_layer(parsed_features)

        return model(transformed_features)

    return serve_tf_examples_fn


def _input_fn(file_pattern, data_accessor, tf_transform_output, batch_size=200):
    """Generates features and label for tuning/training.

    Args:
      file_pattern: List of paths or patterns of input tfrecord files.
      data_accessor: DataAccessor for converting input to RecordBatch.
      tf_transform_output: A TFTransformOutput.
      batch_size: representing the number of consecutive elements of returned
        dataset to combine in a single batch

    Returns:
      A dataset that contains (features, indices) tuple where features is a
        dictionary of Tensors, and indices is a single Tensor of label indices.
    """
    return data_accessor.tf_dataset_factory(
        file_pattern,
        dataset_options.TensorFlowDatasetOptions(
            batch_size=batch_size,
            label_key=t('relative_demand')),
        tf_transform_output.transformed_metadata.schema).repeat()


def _build_keras_model():
    """Creates a DNN Keras model for classifying taxi data.

    Args:
    hidden_units: [int], the layer sizes of the DNN (input layer first).
    learning_rate: [float], learning rate of the Adam optimizer.

    Returns:
    A keras Model.
    """
    real_valued_columns = [
        tf.feature_column.numeric_column(t('avg_total_per_trip_prev4h_area')),
        tf.feature_column.numeric_column(t('avg_total_per_trip_prev4h_city')),
        tf.feature_column.numeric_column(t('avg_ntrips_prev_4h_area')),
        tf.feature_column.numeric_column(t('avg_ntrips_prev_4h_city'))
    ]

    categorical_columns = [
        tf.feature_column.categorical_column_with_identity(t('avg_total_per_trip_prev4h_area'), 5),
        tf.feature_column.categorical_column_with_identity(t('avg_total_per_trip_prev4h_city'), 5),
        tf.feature_column.categorical_column_with_identity(t('avg_ntrips_prev_4h_area'), 100),
        tf.feature_column.categorical_column_with_identity(t('avg_ntrips_prev_4h_city'), 100),
        tf.feature_column.categorical_column_with_identity(t('hour24'), 4),
        tf.feature_column.categorical_column_with_identity(t('area'), 77),
        tf.feature_column.categorical_column_with_identity(t('is_holiday'), 2),
        tf.feature_column.categorical_column_with_identity(t('day_of_week'), 7),
        tf.feature_column.categorical_column_with_identity(t('month'), 12),
        tf.feature_column.categorical_column_with_identity(t('day'), 31),
        tf.feature_column.categorical_column_with_identity(t('hour12'), 12),
        tf.feature_column.categorical_column_with_identity(t('day_period'), 2),
    ]

    indicator_columns = [
        tf.feature_column.indicator_column(categorical_column)
        for categorical_column in categorical_columns
    ]

    crossed_columns = [
        tf.feature_column.crossed_column(
            [
                tf.feature_column.categorical_column_with_identity(t('is_holiday'), 2),
                tf.feature_column.categorical_column_with_identity(t('day_of_week'), 7)
            ],
            hash_bucket_size=14
        ),
        tf.feature_column.crossed_column(
            [
                tf.feature_column.categorical_column_with_identity(t('hour12'), 12),
                tf.feature_column.categorical_column_with_identity(t('day_period'), 2),
            ],
            hash_bucket_size=24
        ),
    ]
    wide_columns = indicator_columns + crossed_columns

    mixed_columns = [
        tf.feature_column.embedding_column(t('area'), 4),
        tf.feature_column.embedding_column(t('month'), 2),
        tf.feature_column.embedding_column(t('day'), 3),
    ]

    model = _wide_and_deep_classifier_advanced(
        wide_columns=wide_columns,
        deep_columns=real_valued_columns,
        mixed_columns=mixed_columns
    )
    return model


def _wide_and_deep_classifier_baseline(wide_columns, deep_columns):
    input_layers = {
        colname: tf.keras.layers.Input(name=colname, shape=(), dtype=tf.float32)
        for colname in features.transformed_names(features.DENSE_FLOAT_FEATURE_KEYS)
    }
    input_layers.update({
        colname: tf.keras.layers.Input(name=colname, shape=(), dtype='int32')
        for colname in features.transformed_names(features.BUCKET_FEATURE_KEYS)
    })
    input_layers.update({
        colname: tf.keras.layers.Input(name=colname, shape=(), dtype='int32') for
        colname in features.transformed_names(features.CATEGORICAL_FEATURE_KEYS)
    })

    deep = tf.keras.layers.DenseFeatures(deep_columns)(input_layers)
    for numnodes in constants.HIDDEN_UNITS:
        deep = tf.keras.layers.Dense(numnodes)(deep)
    wide = tf.keras.layers.DenseFeatures(wide_columns)(input_layers)

    output = tf.keras.layers.Dense(
        1, activation='sigmoid')(
        tf.keras.layers.concatenate([deep, wide]))
    output = tf.squeeze(output, -1)

    model = tf.keras.Model(input_layers, output)
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(lr=constants.LEARNING_RATE),
        metrics=[tf.keras.metrics.BinaryAccuracy()])
    model.summary(print_fn=logging.info)
    return model


def _wide_and_deep_classifier_advanced(wide_columns, deep_columns, mixed_columns):
    input_layers = {
        feature.name: tf.keras.layers.Input(name=feature.name, shape=(), dtype=feature.dtype)
        for feature in wide_columns
    }

    deep = tf.keras.layers.DenseFeatures(deep_columns)(input_layers)
    for numnodes in constants.HIDDEN_UNITS_ADVANCED:
        deep = tf.keras.layers.Dense(numnodes, activation='relu')(deep)

    mix = tf.keras.layers.DenseFeatures(mixed_columns)(input_layers)
    for numnodes in constants.HIDDEN_UNITS_ADVANCED2:
        mix = tf.keras.layers.Dense(numnodes, activation='relu')(mix)

    wide = tf.keras.layers.DenseFeatures(wide_columns)(input_layers)
    for numnodes in constants.HIDDEN_UNITS_ADVANCED_SINK:
        widesink = tf.keras.layers.Dense(numnodes, activation='relu')(wide)

    output = tf.keras.layers.Dense(1)(tf.keras.layers.concatenate([deep, mix, widesink, wide]))
    output = tf.squeeze(output, -1)

    model = tf.keras.Model(input_layers, output)
    model.compile(
        loss=tf.keras.losses.Huber(),
        optimizer=tf.keras.optimizers.Adam(lr=constants.LEARNING_RATE),
        metrics=[
            tf.keras.metrics.LogCoshError(),
            tf.keras.metrics.MeanSquaredLogarithmicError(),
            tf.keras.metrics.MeanAbsolutePercentageError()
        ]
    )
    model.summary(print_fn=logging.info)
    return model


# TFX Trainer will call this function.
def run_fn(fn_args):
    """Train the model based on given args.

    Args:
      fn_args: Holds args used to train the model as name/value pairs.
    """

    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    train_dataset = _input_fn(fn_args.train_files, fn_args.data_accessor,
                              tf_transform_output, constants.TRAIN_BATCH_SIZE)
    eval_dataset = _input_fn(fn_args.eval_files, fn_args.data_accessor,
                             tf_transform_output, constants.EVAL_BATCH_SIZE)

    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        model = _build_keras_model()

    # Write logs to path
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=fn_args.model_run_dir, update_freq='batch')

    model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps
        # callbacks=[tensorboard_callback]
    )

    signatures = {
        'serving_default':
            _get_serve_tf_examples_fn(model,
                                      tf_transform_output).get_concrete_function(
                tf.TensorSpec(
                    shape=[None],
                    dtype=tf.string,
                    name='examples')),
    }
    model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)
