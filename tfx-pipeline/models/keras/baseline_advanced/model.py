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

    Returns:
    A keras Model.
    """

    # Data Input
    sparse = dict(
        avg_total_per_trip_prev4h_area=tf.feature_column.categorical_column_with_identity(
            t('avg_total_per_trip_prev4h_area'), 5),
        avg_total_per_trip_prev4h_city=tf.feature_column.categorical_column_with_identity(
            t('avg_total_per_trip_prev4h_city'), 5),
        avg_ntrips_prev_4h_area=tf.feature_column.categorical_column_with_identity(t('avg_ntrips_prev_4h_area'), 100),
        avg_ntrips_prev_4h_city=tf.feature_column.categorical_column_with_identity(t('avg_ntrips_prev_4h_city'), 100),
        hour24=tf.feature_column.categorical_column_with_identity(t('hour24'), 4),
        area=tf.feature_column.categorical_column_with_identity(t('area'), 77),
        is_holiday=tf.feature_column.categorical_column_with_identity(t('is_holiday'), 2),
        day_of_week=tf.feature_column.categorical_column_with_identity(t('day_of_week'), 7),
        month=tf.feature_column.categorical_column_with_identity(t('month'), 12),
        day=tf.feature_column.categorical_column_with_identity(t('day'), 31),
        hour12=tf.feature_column.categorical_column_with_identity(t('hour12'), 12),
        day_period=tf.feature_column.categorical_column_with_identity(t('day_period'), 2)
    )

    # Categorical Input
    inputs = {
        colname : tf.keras.layers.Input(name=colname, shape=(), dtype='string')
              for colname in sparse.keys()
    }

    # Feature Engineering
    sparse.update(
        is_holiday_day_of_week=tf.feature_column.crossed_column(
            [
                tf.feature_column.categorical_column_with_identity(t('is_holiday'), 2),
                tf.feature_column.categorical_column_with_identity(t('day_of_week'), 7)
            ],
            hash_bucket_size=14
        ),
        hour12_day_period=tf.feature_column.crossed_column(
            [
                tf.feature_column.categorical_column_with_identity(t('hour12'), 12),
                tf.feature_column.categorical_column_with_identity(t('day_period'), 2),
            ],
            hash_bucket_size=24
        )
    )

    embed = dict(
        area=tf.feature_column.embedding_column(sparse['area'], 4),
        month=tf.feature_column.embedding_column(sparse['month'], 2),
        day=tf.feature_column.embedding_column(sparse['day'], 3),
    )

    # one-hot encode the sparse columns
    sparse = {
        colname: tf.feature_column.indicator_column(col)
        for colname, col in sparse.items()
    }

    real_valued_columns = [
        tf.feature_column.numeric_column('avg_total_per_trip_prev4h_area'),
        tf.feature_column.numeric_column('avg_total_per_trip_prev4h_city'),
        tf.feature_column.numeric_column('avg_ntrips_prev_4h_area'),
        tf.feature_column.numeric_column('avg_ntrips_prev_4h_city')
    ]

    return _wide_and_deep_classifier_advanced(
        inputs=inputs,
        wide_columns=sparse.values(),
        deep_columns=real_valued_columns,
        mixed_columns=embed.values()
    )


def _wide_and_deep_classifier_baseline(inputs, wide_columns, deep_columns):
    input_layers = {
        feature.name: tf.keras.layers.Input(name=t(feature.name), shape=())
        for feature in inputs
    }

    deep = tf.keras.layers.DenseFeatures(deep_columns)(input_layers)
    for numnodes in constants.HIDDEN_UNITS:
        deep = tf.keras.layers.Dense(numnodes)(deep)

    wide = tf.keras.layers.DenseFeatures(wide_columns)(input_layers)

    output = tf.keras.layers.concatenate([deep, wide])
    output = tf.keras.layers.Dense(1, activation='sigmoid')(output)
    output = tf.squeeze(output, -1)

    model = tf.keras.Model(input_layers, output)
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(lr=constants.LEARNING_RATE),
        metrics=[tf.keras.metrics.BinaryAccuracy()])
    model.summary(print_fn=logging.info)
    return model


def _wide_and_deep_classifier_advanced(inputs, wide_columns, deep_columns, mixed_columns):
    deep = tf.keras.layers.DenseFeatures(deep_columns)(inputs)
    for numnodes in constants.HIDDEN_UNITS_ADVANCED:
        deep = tf.keras.layers.Dense(numnodes, activation='relu')(deep)

    mix = tf.keras.layers.DenseFeatures(mixed_columns)(inputs)
    for numnodes in constants.HIDDEN_UNITS_ADVANCED2:
        mix = tf.keras.layers.Dense(numnodes, activation='relu')(mix)

    wide = tf.keras.layers.DenseFeatures(wide_columns)(inputs)
    for numnodes in constants.HIDDEN_UNITS_ADVANCED_SINK:
        widesink = tf.keras.layers.Dense(numnodes, activation='relu')(wide)

    output = tf.keras.layers.concatenate([deep, mix, widesink, wide])
    output = tf.keras.layers.Dense(1)(output)
    output = tf.squeeze(output, -1)

    model = tf.keras.Model(inputs, output)
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
