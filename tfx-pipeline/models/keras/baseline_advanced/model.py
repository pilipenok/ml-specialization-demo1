from __future__ import division
from __future__ import print_function

from typing import List

from absl import logging
import tensorflow as tf
import tensorflow_transform as tft

from models.keras.baseline_advanced import constants

from tensorflow_metadata.proto.v0 import schema_pb2
from tfx import v1 as tfx
from tfx_bsl.public import tfxio
from tensorflow_transform.tf_metadata import schema_utils


from models.features import FEATURE_SPEC, LABEL_KEY, FEATURE_KEYS
from pipeline import configs


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


def _input_fn(file_pattern: List[str],
              data_accessor: tfx.components.DataAccessor,
              schema: schema_pb2.Schema,
              batch_size: int) -> tf.data.Dataset:
    """Generates features and label for training.

    Args:
      file_pattern: List of paths or patterns of input tfrecord files.
      data_accessor: DataAccessor for converting input to RecordBatch.
      schema: schema of the input data.
      batch_size: representing the number of consecutive elements of returned
        dataset to combine in a single batch

    Returns:
      A dataset that contains (features, indices) tuple where features is a
        dictionary of Tensors, and indices is a single Tensor of label indices.
    """
    return data_accessor.tf_dataset_factory(
        file_pattern,
        tfxio.TensorFlowDatasetOptions(batch_size=batch_size, label_key=LABEL_KEY),
        schema=schema
    ).repeat()


def _build_keras_model() -> tf.keras.Model:
    """Creates a DNN Keras model for classifying taxi data.

    Returns:
    A keras Model.
    """
    
    sparse = dict(
        hour24=tf.feature_column.categorical_column_with_hash_bucket('hour24', 4, dtype=tf.int64),
        area=tf.feature_column.categorical_column_with_hash_bucket('area', 77, dtype=tf.int64),
        is_holiday=tf.feature_column.categorical_column_with_vocabulary_list('is_holiday', ['true', 'false'], dtype=tf.string),
        day_of_week=tf.feature_column.categorical_column_with_vocabulary_list('day_of_week', range(7), dtype=tf.int64),
        month=tf.feature_column.categorical_column_with_vocabulary_list('month', range(12), dtype=tf.int64),
        day=tf.feature_column.categorical_column_with_vocabulary_list('day', range(31), dtype=tf.int64),
        hour12=tf.feature_column.categorical_column_with_vocabulary_list('hour12', range(12), dtype=tf.int64),
        day_period=tf.feature_column.categorical_column_with_vocabulary_list('day_period', ['am', 'pm'], dtype=tf.string)
    )
    
    real_valued_columns = dict(
        avg_total_per_trip_prev4h_area=tf.feature_column.numeric_column('avg_total_per_trip_prev4h_area'),
        avg_total_per_trip_prev4h_city=tf.feature_column.numeric_column('avg_total_per_trip_prev4h_city'),
        avg_ntrips_prev_4h_area=tf.feature_column.numeric_column('avg_ntrips_prev_4h_area'),
        avg_ntrips_prev_4h_city=tf.feature_column.numeric_column('avg_ntrips_prev_4h_city')
    )

    # Feature Engineering
    sparse.update(
        is_holiday_day_of_week=tf.feature_column.crossed_column([sparse['is_holiday'], sparse['day_of_week']], 2*7),
        hour12_day_period=tf.feature_column.crossed_column([sparse['hour12'],sparse['day_period']], 12*2)
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

    return _wide_and_deep_classifier_baseline(
        wide=real_valued_columns,
        deep=sparse,
        mix=embed
    )

def _wide_and_deep_classifier_baseline(wide, deep, mix):
    inputs = {f: tf.keras.layers.Input(name=f, shape=(), dtype=FEATURE_SPEC[f].dtype) for f in FEATURE_KEYS}

    deep = tf.keras.layers.DenseFeatures(deep.values())(inputs)
    for numnodes in constants.HIDDEN_UNITS_ADVANCED:
        deep = tf.keras.layers.Dense(numnodes, activation='relu')(deep)
        
    mix = tf.keras.layers.DenseFeatures(mix.values())(inputs)
    for numnodes in constants.HIDDEN_UNITS_ADVANCED2:
        mix = tf.keras.layers.Dense(numnodes, activation='relu')(mix)

    wide = tf.keras.layers.DenseFeatures(wide.values())(inputs)
    for numnodes in constants.HIDDEN_UNITS_ADVANCED_SINK:
        wide = tf.keras.layers.Dense(numnodes, activation='relu')(wide)
        
    x = tf.keras.layers.concatenate([deep, wide])
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    x = tf.squeeze(x, -1)
    outputs = x
    
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        loss=tf.keras.losses.Huber(),
        optimizer=tf.keras.optimizers.Adam(lr=constants.LEARNING_RATE),
        metrics=[
            #'accuracy',
            tf.keras.metrics.LogCoshError(),
            tf.keras.metrics.MeanSquaredLogarithmicError(),
            tf.keras.metrics.MeanAbsolutePercentageError()
        ]
    )
    #model.summary(print_fn=logging.info)
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
    output = tf.keras.layers.Dense(1, activation='sigmoid')(output)
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
def run_fn(fn_args: tfx.components.FnArgs):
    """Train the model based on given args.

    Args:
    fn_args: Holds args used to train the model as name/value pairs.
    """

    #tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    # This schema is usually either an output of SchemaGen or a manually-curated
    # version provided by pipeline author. A schema can also derived from TFT
    # graph if a Transform component is used. In the case when either is missing,
    # `schema_from_feature_spec` could be used to generate schema from very simple
    # feature_spec, but the schema returned would be very primitive.

    schema = schema_utils.schema_from_feature_spec(FEATURE_SPEC)

    train_dataset = _input_fn(fn_args.train_files, fn_args.data_accessor, schema, configs.TRAIN_BATCH_SIZE)
    eval_dataset = _input_fn(fn_args.eval_files, fn_args.data_accessor, schema, configs.EVAL_BATCH_SIZE)

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
        validation_steps=fn_args.eval_steps,
        callbacks=[tensorboard_callback]
    )

    # signatures = {
    #     'serving_default':
    #         _get_serve_tf_examples_fn(model, tf_transform_output).get_concrete_function(
    #             tf.TensorSpec(shape=[None],dtype=tf.string,name='examples')),
    # }
    model.save(fn_args.serving_model_dir, save_format='tf')
