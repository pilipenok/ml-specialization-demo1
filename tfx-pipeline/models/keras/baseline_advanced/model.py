from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
from typing import Text, List
from absl import logging

from tfx import v1 as tfx
from tfx_bsl.public import tfxio
import tensorflow as tf
from tensorflow_metadata.proto.v0 import schema_pb2
import tensorflow_transform as tft
from tensorflow_transform.tf_metadata import schema_utils
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

# from pipeline import components

# constants
BASELINE = True  # False #
TASK = 'regr'  # 'class' #

REGULARIZE = False  # True #
DROPOUT = False  # True #

NUM_CLASSES = 11
EPOCHS = 25
TRAIN_BATCH_SIZE = 16
TRAIN_NUM_STEPS = 50000
EVAL_BATCH_SIZE = 16
EVAL_NUM_STEPS = 1000
ES_PATIENCE = 0 if BASELINE else 3

LABEL_KEY = 'trips_bucket' # if TASK == 'class' else 'trips_bucket_num'  # 'n_trips'  # 'log_n_trips' #

MODEL_NAME = f"{LABEL_KEY}-" \
             f"{'baseline' if BASELINE else 'advanced'}" \
             f"{'_regul' if REGULARIZE and not BASELINE else ''}" \
             f"{'_drop' if DROPOUT and not BASELINE else ''}" \
             f"-{EPOCHS}-{TRAIN_BATCH_SIZE}"

LEARNING_RATE = 0.001

HIDDEN_UNITS_BASE_DEEP = [32, 16, 8]
HIDDEN_UNITS_BASE_CONCAT = [32, NUM_CLASSES if TASK == 'class' else 1]

HIDDEN_UNITS_ADV_DEEP = [64, 32, 16]
HIDDEN_UNITS_ADV_EMBED = [8]
HIDDEN_UNITS_ADV_MIX = [16, 4]
HIDDEN_UNITS_ADV_WIDE = [512, 64, 4]
HIDDEN_UNITS_ADV_CONCAT = [4, NUM_CLASSES if TASK == 'class' else 1]

# features
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

# Name of features which have string values and are mapped to integers.
VOCAB_FEATURE_KEYS = []
# Number of vocabulary terms used for encoding VOCAB_FEATURES by tf.transform
VOCAB_SIZE = 2
# Count of out-of-vocab buckets in which unrecognized VOCAB_FEATURES are hashed.
OOV_SIZE = 10


def transformed_name(key: Text) -> Text:
    """Generate the name of the transformed feature from original name."""
    return key + '_xf'


def vocabulary_name(key: Text) -> Text:
    """Generate the name of the vocabulary feature from original name."""
    return key + '_vocab'


def transformed_names(keys: List[Text]) -> List[Text]:
    """Transform multiple feature names at once."""
    return [transformed_name(key) for key in keys]


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
    LABEL_KEY: tf.io.FixedLenFeature(
        shape=[1 if '_num' in LABEL_KEY else NUM_CLASSES],
        dtype=tf.float32 if '_num' in LABEL_KEY else tf.int64
    ),
    'year': tf.io.FixedLenFeature(shape=[1], dtype=tf.int64),
}


def get_schema():
    return schema_utils.schema_from_feature_spec(FEATURE_SPEC)
    # return components.schema_gen().outputs['schema']


# model
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
        area=categorical_column_with_identity('area', 78),
        quarter=categorical_column_with_vocabulary_list('quarter', range(1, 4), dtype=tf.int64),
        month=categorical_column_with_vocabulary_list('month', range(1, 13), dtype=tf.int64),
        day=categorical_column_with_vocabulary_list('day', range(1, 32), dtype=tf.int64),
        hour=categorical_column_with_identity('hour', 24),
        day_period=categorical_column_with_vocabulary_list('day_period', ['am', 'pm'], dtype=tf.string),
        week=categorical_column_with_identity('week', 54),
        day_of_week=categorical_column_with_vocabulary_list('day_of_week', range(1, 8), dtype=tf.int64),
        is_weekend=categorical_column_with_vocabulary_list('is_weekend', ['true', 'false'], dtype=tf.string),
        is_holiday=categorical_column_with_vocabulary_list('is_holiday', ['true', 'false'], dtype=tf.string),
    )
    sparse.update(
        is_holiday_day_of_week=crossed_column([sparse['is_holiday'], sparse['day_of_week']], 2 * 7),
    )

    embed = dict(
        area_emb=embedding_column(sparse['area'], 4),
        quarter_emb=embedding_column(sparse['quarter'], 2),
        month_emb=embedding_column(sparse['month'], 2),
        day_emb=embedding_column(sparse['day'], 3),
        week_emb=embedding_column(sparse['week'], 4),
        day_of_week_emb=embedding_column(sparse['day_of_week'], 2),
    )

    # one-hot encode the sparse columns
    sparse = {
        colname: indicator_column(col)
        for colname, col in sparse.items()
    }

    if BASELINE:
        return _wide_and_deep_classifier_baseline(
            deep={**real_valued_columns, **embed},
            wide=sparse,
        )
    else:
        return _wide_and_deep_classifier_advanced(
            deep=real_valued_columns,
            embed=embed,
            wide=sparse,
            regularizer=REGULARIZE,
            dropout=DROPOUT
        )


def _wide_and_deep_classifier_baseline(deep, wide):
    deep_idx, concat_idx = 0, 0

    inputs = {f: Input(name=f, shape=(), dtype=FEATURE_SPEC[f].dtype) for f in FEATURE_KEYS}

    deep = DenseFeatures(deep.values(), name='deep_inputs')(inputs)
    for numnodes in HIDDEN_UNITS_BASE_DEEP:
        deep_idx += 1
        deep = Dense(numnodes, activation='relu', name='deep_' + str(deep_idx))(deep)

    wide = DenseFeatures(wide.values(), name='wide_inputs')(inputs)

    concat = Concatenate()([deep, wide])
    for numnodes in HIDDEN_UNITS_BASE_CONCAT[:-1]:
        concat_idx += 1
        concat = Dense(numnodes, activation='sigmoid', name='concat_' + str(concat_idx))(concat)

    outputs = Dense(HIDDEN_UNITS_BASE_CONCAT[-1], name='model_output')(concat)

    model = tf.keras.Model(inputs, outputs)
    return model


def _wide_and_deep_classifier_advanced(deep, embed, wide, regularizer=False, dropout=False):
    deep_idx, embed_idx, wide_idx, mix_idx, concat_idx = 0, 0, 0, 0, 0

    inputs = {f: Input(name=f, shape=(), dtype=FEATURE_SPEC[f].dtype) for f in FEATURE_KEYS}

    deep = DenseFeatures(deep.values(), name='deep_inputs')(inputs)
    for numnodes in HIDDEN_UNITS_ADV_DEEP:
        deep_idx += 1
        deep = Dense(
            numnodes, activation='tanh',
            kernel_regularizer=l1_l2(l1=0.01, l2=0.01) if regularizer else None,
            name='deep_' + str(deep_idx)
        )(deep)
        if dropout:
            deep = Dropout(0.5, name=f"dropout_deep_{deep_idx}")(deep)

    embed = DenseFeatures(embed.values(), name='embed_inputs')(inputs)
    for numnodes in HIDDEN_UNITS_ADV_EMBED:
        embed_idx += 1
        embed = Dense(
            numnodes, activation='relu',
            kernel_regularizer=l1_l2(l1=0.01, l2=0.01) if regularizer else None,
            name='embed_' + str(embed_idx)
        )(embed)
        if dropout:
            embed = Dropout(0.5, name=f"dropout_embed_{embed_idx}")(embed)

    mix = Concatenate()([deep, embed])
    for numnodes in HIDDEN_UNITS_ADV_MIX:
        mix_idx += 1
        mix = Dense(
            numnodes, activation='tanh',
            kernel_regularizer=l1_l2(l1=0.01, l2=0.01) if regularizer else None,
            name='mix_' + str(mix_idx)
        )(mix)
        if dropout:
            mix = Dropout(0.5, name=f"dropout_mix_{mix_idx}")(mix)

    wide = DenseFeatures(wide.values(), name='wide_inputs')(inputs)
    for numnodes in HIDDEN_UNITS_ADV_WIDE:
        wide_idx += 1
        wide = Dense(
            numnodes, activation='relu',
            kernel_regularizer=l1_l2(l1=0.01, l2=0.01) if regularizer else None,
            name='wide_' + str(wide_idx)
        )(wide)
        if dropout:
            wide = Dropout(0.5, name=f"dropout_wide_{wide_idx}")(wide)

    concat = Concatenate()([wide, mix])
    for numnodes in HIDDEN_UNITS_ADV_CONCAT[:-1]:
        concat_idx += 1
        concat = Dense(numnodes, activation='tanh', name='concat_' + str(concat_idx))(concat)

    outputs = Dense(HIDDEN_UNITS_ADV_CONCAT[-1], name='model_output')(concat)

    model = tf.keras.Model(inputs, outputs)
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
        if TASK == 'class':
            loss = loss_sce(from_logits=True)
            metrics = [
                Accuracy(),
                AUC(curve='ROC', name='ROC'),
                AUC(curve='PR', name='PR'),
                SparseCategoricalAccuracy(),
                SparseCategoricalCrossentropy(from_logits=True)
            ]
        elif TASK == 'regr':
            loss = loss_mse() if BASELINE else loss_mape()
            metrics = [
                MeanSquaredError(),
                MeanAbsolutePercentageError(),
            ]

        model = _build_keras_model()
        model.compile(
            loss=loss,
            optimizer=tf.keras.optimizers.Adam(lr=LEARNING_RATE),
            metrics=metrics
        )
    model.summary(print_fn=logging.info)

    earlystopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_mean_squared_error' if BASELINE else 'val_mean_absolute_percentage_error',
        patience=ES_PATIENCE,
        restore_best_weights=True,
        verbose=1
    )
    # Write logs to path
    tb_logdir = os.path.join(
        fn_args.model_run_dir[:fn_args.model_run_dir.rfind('/')],
        f"{MODEL_NAME}-({fn_args.model_run_dir[fn_args.model_run_dir.rfind('/') + 1:]})"
    )
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=tb_logdir,
        update_freq=10,
        histogram_freq=1,
        embeddings_freq=1,
    )

    model.fit(
        train_dataset,
        epochs=EPOCHS,
        steps_per_epoch=TRAIN_NUM_STEPS,
        validation_data=eval_dataset,
        validation_steps=EVAL_NUM_STEPS,
        callbacks=[
            earlystopping_callback,
            tensorboard_callback,
        ]
    )

    if BASELINE:
        logging.info("Baseline DNN architecture:\n"
                     f"\tHIDDEN_UNITS_BASE_DEEP = {HIDDEN_UNITS_BASE_DEEP}\n"
                     f"\tHIDDEN_UNITS_BASE_CONCAT = {HIDDEN_UNITS_BASE_CONCAT}"
                     )
    else:
        logging.info(f"Advanced DNN architecture ("
                     f"{'with' if REGULARIZE else 'without'} regularization, "
                     f"{'with' if DROPOUT else 'without'} dropout):\n"
                     f"\tHIDDEN_UNITS_ADV_DEEP = {HIDDEN_UNITS_ADV_DEEP}\n"
                     f"\tHIDDEN_UNITS_ADV_EMBED = {HIDDEN_UNITS_ADV_EMBED}\n"
                     f"\tHIDDEN_UNITS_ADV_MIX = {HIDDEN_UNITS_ADV_MIX}\n"
                     f"\tHIDDEN_UNITS_ADV_WIDE = {HIDDEN_UNITS_ADV_WIDE}\n"
                     f"\tHIDDEN_UNITS_ADV_CONCAT = {HIDDEN_UNITS_ADV_CONCAT}"
                     )
    logging.info(f"TensorBoard log directory: {tb_logdir}")

    # signatures = {
    #     'serving_default':
    #         _get_serve_tf_examples_fn(model, tf_transform_output).get_concrete_function(
    #             tf.TensorSpec(shape=[None],dtype=tf.string,name='examples')),
    # }
    model.save(fn_args.serving_model_dir, save_format='tf')
