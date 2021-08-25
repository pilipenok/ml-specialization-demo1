from __future__ import division
from __future__ import print_function

from typing import List
from absl import logging

import tensorflow as tf
from tfx import v1 as tfx
import tensorflow_transform as tft

from tensorflow_metadata.proto.v0 import schema_pb2
from tfx_bsl.public import tfxio
from tensorflow import feature_column as tfc
from tensorflow.keras.layers import Input, DenseFeatures, Dense, Concatenate

from tensorflow_transform.tf_metadata import schema_utils

HIDDEN_UNITS_DEEP_TANH = [64, 32, 16]
HIDDEN_UNITS_DEEP_RELU = [16, 8, 4]
HIDDEN_UNITS_WIDE = [512, 64, 4]
HIDDEN_UNITS_MIX = [128, 32, 2]
HIDDEN_UNITS = [4, 1]

LEARNING_RATE = 0.001

TRAIN_BATCH_SIZE = 40
EVAL_BATCH_SIZE = 40

FEATURE_KEYS = \
    "area,year," \
    "quarter,quarter_num,quarter_cos,quarter_sin," \
    "month,month_num,month_cos,month_sin," \
    "day,day_num,day_cos,day_sin," \
    "hour,hour_num,hour_cos,hour_sin," \
    "day_period," \
    "week,week_num,week_cos,week_sin," \
    "day_of_week,day_of_week_num,day_of_week_cos,day_of_week_sin," \
    "weekday_hour_num,weekday_hour_cos,weekday_hour_sin," \
    "yearday_hour_num,yearday_hour_cos,yearday_hour_sin," \
    "is_weekend,is_holiday" \
        .split(',')

# Name of features which have continuous float values. These features will be
# used as their own values.
DENSE_FLOAT_FEATURE_KEYS = \
    "year," \
    "quarter_num,quarter_cos,quarter_sin," \
    "month_num,month_cos,month_sin," \
    "day_num,day_cos,day_sin," \
    "hour_num,hour_cos,hour_sin," \
    "week_num,week_cos,week_sin," \
    "day_of_week_num,day_of_week_cos,day_of_week_sin," \
    "weekday_hour_num,weekday_hour_cos,weekday_hour_sin," \
    "yearday_hour_num,yearday_hour_cos,yearday_hour_sin" \
        .split(',')

# Name of features which have continuous float values. These features will be
# bucketized using `tft.bucketize`, and will be used as categorical features.
BUCKET_FEATURE_KEYS = "area,quarter,month,day,hour,week,day_of_week".split(',')
# Number of buckets used by tf.transform for encoding each feature. The length
# of this list should be the same with BUCKET_FEATURE_KEYS.
BUCKET_FEATURE_BUCKET_COUNT = [77, 4, 12, 31, 12, 53, 7]

# Name of features which have categorical values which are mapped to integers.
# These features will be used as categorical features.
CATEGORICAL_FEATURE_KEYS = "day_period,is_weekend,is_holiday".split(',')
# Number of buckets to use integer numbers as categorical features. The length
# of this list should be the same with CATEGORICAL_FEATURE_KEYS.
CATEGORICAL_FEATURE_MAX_VALUES = [2, 2, 2]

LABEL_KEY = 'log_n_trips'  # 'n_trips' #

# Since we're not generating or creating a schema, we will instead create a feature spec.
FEATURE_SPEC = {
    **{
        feature: tf.io.FixedLenFeature(shape=[1], dtype=tf.float32)
        for feature in DENSE_FLOAT_FEATURE_KEYS
    },
    **{
        feature: tf.io.FixedLenFeature(shape=[1], dtype=tf.int64)
        for feature in BUCKET_FEATURE_KEYS
    },
    **{
        feature: tf.io.FixedLenFeature(shape=[1], dtype=tf.string)
        for feature in CATEGORICAL_FEATURE_KEYS
    },
    LABEL_KEY: tf.io.FixedLenFeature(shape=[1], dtype=tf.float32)
}
FEATURE_SPEC['year'] = tf.io.FixedLenFeature(shape=[1], dtype=tf.int64)


def get_schema():
    return schema_utils.schema_from_feature_spec(FEATURE_SPEC)
    # return components.schema_gen().outputs['schema']


def preprocessing_fn(inputs):
    """tf.transform's callback function for preprocessing inputs.

    Args:
        inputs: map from feature keys to raw not-yet-transformed features.

    Returns:
        Map from string feature key to transformed feature operations.
    """
    
    outputs = inputs

    return outputs    

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

    real_valued_columns = {
        feature: tfc.numeric_column(feature)
        for feature in DENSE_FLOAT_FEATURE_KEYS
    }

    sparse = dict(
        area=tfc.categorical_column_with_identity('area', 78),
        quarter=tfc.categorical_column_with_vocabulary_list('quarter', range(1, 4), dtype=tf.int64),
        month=tfc.categorical_column_with_vocabulary_list('month', range(1, 13), dtype=tf.int64),
        day=tfc.categorical_column_with_vocabulary_list('day', range(1, 32), dtype=tf.int64),
        hour=tfc.categorical_column_with_identity('hour', 24),
        day_period=tfc.categorical_column_with_vocabulary_list('day_period', ['am', 'pm'], dtype=tf.string),
        week=tfc.categorical_column_with_identity('week', 54),
        day_of_week=tfc.categorical_column_with_vocabulary_list('day_of_week', range(1, 8), dtype=tf.int64),
        is_weekend=tfc.categorical_column_with_vocabulary_list('is_weekend', ['true', 'false'], dtype=tf.string),
        is_holiday=tfc.categorical_column_with_vocabulary_list('is_holiday', ['true', 'false'], dtype=tf.string),
    )

    # Feature Engineering
    sparse.update(
        #         hour_bucket = categorical_column_with_hash_bucket('hour', 4, dtype=tf.int64),
        is_holiday_day_of_week=tfc.crossed_column([sparse['is_holiday'], sparse['day_of_week']], 2 * 7),
    )
    embed = dict(
        area_emb=tfc.embedding_column(sparse['area'], 4),
        quarter_emb=tfc.embedding_column(sparse['quarter'], 2),
        month_emb=tfc.embedding_column(sparse['month'], 2),
        day_emb=tfc.embedding_column(sparse['day'], 3),
        week_emb=tfc.embedding_column(sparse['week'], 4),
        day_of_week_emb=tfc.embedding_column(sparse['day_of_week'], 2),
    )

    # one-hot encode the sparse columns
    sparse = {
        colname: tfc.indicator_column(col)
        for colname, col in sparse.items()
    }

    return _wide_and_deep_classifier_baseline(
        deep=real_valued_columns,
        wide=sparse,
        mix=embed
    )


def _wide_and_deep_classifier_baseline(deep, wide, mix):
    deep_idx, wide_idx, mix_idx, concat_idx = 0, 0, 0, 0

    inputs = {f: Input(name=f, shape=(), dtype=FEATURE_SPEC[f].dtype) for f in FEATURE_KEYS}

    deep = DenseFeatures(deep.values(), name='deep_inputs')(inputs)
    for numnodes in HIDDEN_UNITS_DEEP_TANH:
        deep_idx += 1
        deep = Dense(numnodes, activation='tanh', name='deep_' + str(deep_idx))(deep)
    for numnodes in HIDDEN_UNITS_DEEP_RELU:
        deep_idx += 1
        deep = Dense(numnodes, activation='relu', name='deep_' + str(deep_idx))(deep)

    wide = DenseFeatures(wide.values(), name='wide_inputs')(inputs)
    for numnodes in HIDDEN_UNITS_WIDE:
        wide_idx += 1
        wide = Dense(numnodes, activation='relu', name='wide_' + str(wide_idx))(wide)

    mix = DenseFeatures(mix.values(), name='mix_inputs')(inputs)
    for numnodes in HIDDEN_UNITS_MIX:
        mix_idx += 1
        mix = Dense(numnodes, activation='relu', name='mix_' + str(mix_idx))(mix)

    x = Concatenate()([deep, wide, mix])
    for numnodes in HIDDEN_UNITS:
        concat_idx += 1
        x = Dense(numnodes, name='concat_' + str(concat_idx))(x)

    try:
        logging.debug(f"output shape of the last dense layer = {x.output_shape()}")
        outputs = tf.squeeze(x, -1, name='model_output')
    except Exception as e:
        logging.error(f"{e.__class__}: {e}")
        outputs = x

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        loss=tf.keras.losses.MeanAbsolutePercentageError(),
        # tf.keras.losses.MeanSquaredError(), # tf.keras.losses.Huber(), #
        optimizer=tf.keras.optimizers.Adam(lr=LEARNING_RATE),
        metrics=[
            # 'accuracy',
            # tf.keras.metrics.LogCoshError(),
            # tf.keras.metrics.MeanSquaredLogarithmicError(),
            tf.keras.metrics.MeanAbsolutePercentageError(),
            tf.keras.metrics.RootMeanSquaredError()
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

    # tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    # This schema is usually either an output of SchemaGen or a manually-curated
    # version provided by pipeline author. A schema can also derived from TFT
    # graph if a Transform component is used. In the case when either is missing,
    # `schema_from_feature_spec` could be used to generate schema from very simple
    # feature_spec, but the schema returned would be very primitive.

    schema = get_schema()

    train_dataset = _input_fn(fn_args.train_files, fn_args.data_accessor, schema, TRAIN_BATCH_SIZE)
    eval_dataset = _input_fn(fn_args.eval_files, fn_args.data_accessor, schema, EVAL_BATCH_SIZE)

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
