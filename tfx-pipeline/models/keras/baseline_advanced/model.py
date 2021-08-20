from __future__ import division
from __future__ import print_function

import os
from typing import List
from absl import logging

from tfx import v1 as tfx
from tfx_bsl.public import tfxio
import tensorflow as tf
from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow.feature_column import \
    numeric_column, \
    categorical_column_with_identity, \
    categorical_column_with_vocabulary_list, \
    categorical_column_with_hash_bucket, \
    crossed_column, \
    embedding_column, \
    indicator_column
from tensorflow.keras.layers import Input, DenseFeatures, Dense, Concatenate, Lambda

from models.keras.baseline_advanced import constants
from models.keras.baseline_advanced.features import LABEL_KEY, FEATURE_KEYS, DENSE_FLOAT_FEATURE_KEYS, FEATURE_SPEC, get_schema


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
        feature: numeric_column(feature) 
        for feature in DENSE_FLOAT_FEATURE_KEYS
    }
    
    sparse = dict(
        area = categorical_column_with_identity('area', 78),
        quarter = categorical_column_with_vocabulary_list('quarter', range(1,4), dtype=tf.int64),
        month = categorical_column_with_vocabulary_list('month', range(1,13), dtype=tf.int64),
        day = categorical_column_with_vocabulary_list('day', range(1,32), dtype=tf.int64),
        hour = categorical_column_with_identity('hour', 24),
        day_period = categorical_column_with_vocabulary_list('day_period', ['am', 'pm'], dtype=tf.string),
        week = categorical_column_with_identity('week', 54),
        day_of_week = categorical_column_with_vocabulary_list('day_of_week', range(1,8), dtype=tf.int64),
        is_weekend = categorical_column_with_vocabulary_list('is_weekend', ['true', 'false'], dtype=tf.string),
        is_holiday = categorical_column_with_vocabulary_list('is_holiday', ['true', 'false'], dtype=tf.string),
    )

    # Feature Engineering
    sparse.update(
#         hour_bucket = categorical_column_with_hash_bucket('hour', 4, dtype=tf.int64),
        is_holiday_day_of_week = crossed_column([sparse['is_holiday'], sparse['day_of_week']], 2*7),
    )
    embed = dict(
        area_emb = embedding_column(sparse['area'], 4),
        quarter_emb = embedding_column(sparse['quarter'], 2),
        month_emb = embedding_column(sparse['month'], 2),
        day_emb = embedding_column(sparse['day'], 3),
        week_emb = embedding_column(sparse['week'], 4),
        day_of_week_emb = embedding_column(sparse['day_of_week'], 2),
    )

    # one-hot encode the sparse columns
    sparse = {
        colname: indicator_column(col)
        for colname, col in sparse.items()
    }
    
    if constants.baseline:
        return _wide_and_deep_classifier_baseline(
            deep=real_valued_columns,
            wide=sparse,
            mix=embed
        )
    else:
        return _wide_and_deep_classifier_advanced(
            deep=real_valued_columns,
            wide=sparse,
            mix=embed
        )


def _wide_and_deep_classifier_baseline(deep, wide, mix):
    deep_idx, wide_idx, mix_idx, concat_idx = 0, 0, 0, 0
    
    inputs = {f: Input(name=f, shape=(), dtype=FEATURE_SPEC[f].dtype) for f in FEATURE_KEYS}

    deep = DenseFeatures(deep.values(), name='deep_inputs')(inputs)
    for numnodes in constants.HIDDEN_UNITS_DEEP_TANH:
        deep_idx += 1
        deep = Dense(numnodes, activation='tanh', name='deep_'+str(deep_idx))(deep)
    for numnodes in constants.HIDDEN_UNITS_DEEP_RELU:
        deep_idx += 1
        deep = Dense(numnodes, activation='relu', name='deep_'+str(deep_idx))(deep)

    wide = DenseFeatures(wide.values(), name='wide_inputs')(inputs)
    for numnodes in constants.HIDDEN_UNITS_WIDE:
        wide_idx += 1
        wide = Dense(numnodes, activation='relu', name='wide_'+str(wide_idx))(wide)
        
    mix = DenseFeatures(mix.values(), name='mix_inputs')(inputs)
    for numnodes in constants.HIDDEN_UNITS_MIX:
        mix_idx += 1
        mix = Dense(numnodes, activation='relu', name='mix_'+str(mix_idx))(mix)
        
    concat = Concatenate()([deep, wide, mix])
    for numnodes in constants.HIDDEN_UNITS_CONCAT:
        concat_idx += 1
        concat = Dense(numnodes, name='concat_'+str(concat_idx))(concat)
    
    outputs = tf.squeeze(concat, -1, name='model_output')
#     outputs = Lambda(lambda x: x, name='model_output')(concat)
    
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        loss=tf.keras.losses.MeanAbsolutePercentageError(), # tf.keras.losses.MeanSquaredError(), # tf.keras.losses.Huber(), # 
        optimizer=tf.keras.optimizers.Adam(lr=constants.LEARNING_RATE),
        metrics=[
            #'accuracy',
            # tf.keras.metrics.LogCoshError(),
            # tf.keras.metrics.MeanSquaredLogarithmicError(),
            tf.keras.metrics.MeanAbsolutePercentageError(),
            tf.keras.metrics.RootMeanSquaredError()
        ]
    )
    model.summary(print_fn=logging.info)
    return model


def _wide_and_deep_classifier_advanced(inputs, wide_columns, deep_columns, mixed_columns):
    deep = DenseFeatures(deep_columns)(inputs)
    for numnodes in constants.HIDDEN_UNITS_ADVANCED:
        deep = Dense(numnodes, activation='relu')(deep)

    mix = DenseFeatures(mixed_columns)(inputs)
    for numnodes in constants.HIDDEN_UNITS_ADVANCED2:
        mix = Dense(numnodes, activation='relu')(mix)

    wide = DenseFeatures(wide_columns)(inputs)
    for numnodes in constants.HIDDEN_UNITS_ADVANCED_SINK:
        widesink = Dense(numnodes, activation='relu')(wide)

    output = concatenate([deep, mix, widesink, wide])
    output = Dense(1, activation='sigmoid')(output)
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

    schema = get_schema()

    train_dataset = _input_fn(fn_args.train_files, fn_args.data_accessor, schema, constants.TRAIN_BATCH_SIZE)
    eval_dataset = _input_fn(fn_args.eval_files, fn_args.data_accessor, schema, constants.EVAL_BATCH_SIZE)
    
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        model = _build_keras_model()

    earlystopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_mean_absolute_percentage_error', 
        patience=constants.ES_PATIENCE,
        restore_best_weights=True, 
        verbose=1
    )
    # Write logs to path
    tb_logdir = os.path.join(
        fn_args.model_run_dir[:fn_args.model_run_dir.rfind('/')], 
        f"{constants.MODEL_NAME} ({fn_args.model_run_dir[fn_args.model_run_dir.rfind('/')+1:]})"
    )
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=tb_logdir,
        update_freq=10,#'batch',#
        histogram_freq=1,
        embeddings_freq=1,
    )

    model.fit(
        train_dataset,
        epochs=constants.EPOCHS,
#         steps_per_epoch=fn_args.train_steps,
        steps_per_epoch=constants.TRAIN_NUM_STEPS,
        validation_data=eval_dataset,
#         validation_steps=fn_args.eval_steps,
        validation_steps=constants.EVAL_NUM_STEPS,
        callbacks=[
            earlystopping_callback,
            tensorboard_callback,
        ]
    )
    
    logging.info("DNN architecture:\n"
                 f"\tHIDDEN_UNITS_DEEP_TANH = {constants.HIDDEN_UNITS_DEEP_TANH}\n"
                 f"\tHIDDEN_UNITS_DEEP_RELU = {constants.HIDDEN_UNITS_DEEP_RELU}\n"
                 f"\tHIDDEN_UNITS_WIDE = {constants.HIDDEN_UNITS_WIDE}\n"
                 f"\tHIDDEN_UNITS_MIX = {constants.HIDDEN_UNITS_MIX}\n"
                 f"\tHIDDEN_UNITS_CONCAT = {constants.HIDDEN_UNITS_CONCAT}"
    )
    logging.info(f"TensorBoard log directory: {tb_logdir}")

    # signatures = {
    #     'serving_default':
    #         _get_serve_tf_examples_fn(model, tf_transform_output).get_concrete_function(
    #             tf.TensorSpec(shape=[None],dtype=tf.string,name='examples')),
    # }
    model.save(fn_args.serving_model_dir, save_format='tf')
