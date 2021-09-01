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
from tensorflow.keras.layers import Input, DenseFeatures, Dense, Concatenate, Dropout
from tensorflow.keras.losses import \
    SparseCategoricalCrossentropy as loss_sce, \
    MeanSquaredError as loss_mse, MeanAbsolutePercentageError as loss_mape
from tensorflow.keras.metrics import \
    Accuracy, AUC, SparseCategoricalAccuracy, SparseCategoricalCrossentropy, \
    MeanSquaredError, MeanAbsolutePercentageError
from tensorflow.keras.regularizers import l1_l2

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
    sparse.update(
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
            deep={**real_valued_columns, **embed},
            wide=sparse,
        )
    else:
        return _wide_and_deep_classifier_advanced(
            deep=real_valued_columns,
            embed=embed,
            wide=sparse,
            regularizer=constants.regularizer,
            dropout=constants.dropout
        )


def _wide_and_deep_classifier_baseline(deep, wide):
    deep_idx, concat_idx = 0, 0
    
    inputs = {f: Input(name=f, shape=(), dtype=FEATURE_SPEC[f].dtype) for f in FEATURE_KEYS}

    deep = DenseFeatures(deep.values(), name='deep_inputs')(inputs)
    for numnodes in constants.HIDDEN_UNITS_BASE_DEEP:
        deep_idx += 1
        deep = Dense(numnodes, activation='relu', name='deep_'+str(deep_idx))(deep)

    wide = DenseFeatures(wide.values(), name='wide_inputs')(inputs)
    
    concat = Concatenate()([deep, wide])
    for numnodes in constants.HIDDEN_UNITS_BASE_CONCAT[:-1]:
        concat_idx += 1
        concat = Dense(numnodes, activation='sigmoid', name='concat_'+str(concat_idx))(concat)
    
    outputs = Dense(constants.HIDDEN_UNITS_BASE_CONCAT[-1], name='model_output')(concat)
    
    model = tf.keras.Model(inputs, outputs)
    return model


def _wide_and_deep_classifier_advanced(deep, embed, wide, regularizer=False, dropout=False):
    deep_idx, embed_idx, wide_idx, mix_idx, concat_idx = 0, 0, 0, 0, 0
    
    inputs = {f: Input(name=f, shape=(), dtype=FEATURE_SPEC[f].dtype) for f in FEATURE_KEYS}

    deep = DenseFeatures(deep.values(), name='deep_inputs')(inputs)
    for numnodes in constants.HIDDEN_UNITS_ADV_DEEP:
        deep_idx += 1
        deep = Dense(
            numnodes, activation='tanh', 
            kernel_regularizer=l1_l2(l1=0.01, l2=0.01) if regularizer else None,
            name='deep_'+str(deep_idx)
        )(deep)
        if dropout:
            deep = Dropout(0.5, name=f"dropout_deep_{deep_idx}")(deep)
        
    embed = DenseFeatures(embed.values(), name='embed_inputs')(inputs)
    for numnodes in constants.HIDDEN_UNITS_ADV_EMBED:
        embed_idx += 1
        embed = Dense(
            numnodes, activation='relu', 
            kernel_regularizer=l1_l2(l1=0.01, l2=0.01) if regularizer else None,
            name='embed_'+str(embed_idx)
        )(embed)
        if dropout:
            embed = Dropout(0.5, name=f"dropout_embed_{embed_idx}")(embed)
    
    mix = Concatenate()([deep, embed])
    for numnodes in constants.HIDDEN_UNITS_ADV_MIX:
        mix_idx += 1
        mix = Dense(
            numnodes, activation='tanh', 
            kernel_regularizer=l1_l2(l1=0.01, l2=0.01) if regularizer else None,
            name='mix_'+str(mix_idx)
        )(mix)
        if dropout:
            mix = Dropout(0.5, name=f"dropout_mix_{mix_idx}")(mix)

    wide = DenseFeatures(wide.values(), name='wide_inputs')(inputs)
    for numnodes in constants.HIDDEN_UNITS_ADV_WIDE:
        wide_idx += 1
        wide = Dense(
            numnodes, activation='relu', 
            kernel_regularizer=l1_l2(l1=0.01, l2=0.01) if regularizer else None,
            name='wide_'+str(wide_idx)
        )(wide)
        if dropout:
            wide = Dropout(0.5, name=f"dropout_wide_{wide_idx}")(wide)
        
    concat = Concatenate()([wide, mix])
    for numnodes in constants.HIDDEN_UNITS_ADV_CONCAT[:-1]:
        concat_idx += 1
        concat = Dense(numnodes, activation='tanh', name='concat_'+str(concat_idx))(concat)
    
    outputs = Dense(constants.HIDDEN_UNITS_ADV_CONCAT[-1], name='model_output')(concat)
    
    model = tf.keras.Model(inputs, outputs)
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
        if constants.task == 'class':
            loss = loss_sce(from_logits=True)
            metrics = [
                Accuracy(),
                AUC(curve='ROC', name='ROC'),
                AUC(curve='PR', name='PR'),
                SparseCategoricalAccuracy(),
                SparseCategoricalCrossentropy(from_logits=True)
            ]
        elif constants.task == 'regr':
            loss = loss_mse() if constants.baseline else loss_mape()
            metrics = [
                MeanSquaredError(),
                MeanAbsolutePercentageError(),
            ]
        
        model = _build_keras_model()
        model.compile(
            loss=loss,
            optimizer=tf.keras.optimizers.Adam(lr=constants.LEARNING_RATE),
            metrics=metrics
        )
    model.summary(print_fn=logging.info)

    earlystopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_mean_squared_error' if constants.baseline else 'val_mean_absolute_percentage_error', 
        patience=constants.ES_PATIENCE,
        restore_best_weights=True, 
        verbose=1
    )
    # Write logs to path
    tb_logdir = os.path.join(
        fn_args.model_run_dir[:fn_args.model_run_dir.rfind('/')], 
        f"{constants.MODEL_NAME}-({fn_args.model_run_dir[fn_args.model_run_dir.rfind('/')+1:]})"
    )
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=tb_logdir,
        update_freq=10,
        histogram_freq=1,
        embeddings_freq=1,
    )

    model.fit(
        train_dataset,
        epochs=constants.EPOCHS,
        steps_per_epoch=constants.TRAIN_NUM_STEPS,
        validation_data=eval_dataset,
        validation_steps=constants.EVAL_NUM_STEPS,
        callbacks=[
            earlystopping_callback,
            tensorboard_callback,
        ]
    )
    
    if constants.baseline:
        logging.info("Baseline DNN architecture:\n"
                     f"\tHIDDEN_UNITS_BASE_DEEP = {constants.HIDDEN_UNITS_BASE_DEEP}\n"
                     f"\tHIDDEN_UNITS_BASE_CONCAT = {constants.HIDDEN_UNITS_BASE_CONCAT}"
        )
    else:
        logging.info(f"Advanced DNN architecture ("
                     f"{'with' if constants.regularizer else 'without'} regularization, "
                     f"{'with' if constants.dropout else 'without'} dropout):\n"
                     f"\tHIDDEN_UNITS_ADV_DEEP = {constants.HIDDEN_UNITS_ADV_DEEP}\n"
                     f"\tHIDDEN_UNITS_ADV_EMBED = {constants.HIDDEN_UNITS_ADV_EMBED}\n"
                     f"\tHIDDEN_UNITS_ADV_MIX = {constants.HIDDEN_UNITS_ADV_MIX}\n"
                     f"\tHIDDEN_UNITS_ADV_WIDE = {constants.HIDDEN_UNITS_ADV_WIDE}\n"
                     f"\tHIDDEN_UNITS_ADV_CONCAT = {constants.HIDDEN_UNITS_ADV_CONCAT}"
        )
    logging.info(f"TensorBoard log directory: {tb_logdir}")

    # signatures = {
    #     'serving_default':
    #         _get_serve_tf_examples_fn(model, tf_transform_output).get_concrete_function(
    #             tf.TensorSpec(shape=[None],dtype=tf.string,name='examples')),
    # }
    model.save(fn_args.serving_model_dir, save_format='tf')
