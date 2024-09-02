import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K


def normalise_series(data):
    max_point, min_point, median_value = data.max(), data.min(), np.median(data)
    return (data - median_value) / abs(max_point - min_point)


def down_to_zero(data: np.array, edge: float) -> np.array:
    # edge = np.std(data)
    # print(edge)
    filter_values = np.vectorize(lambda x: 1.0 if abs(x) > edge else 0.0)
    return filter_values(data)


def get_borders(data: np.array, scale=np.exp(1)):
    loc_max_ind = np.argmax(data)
    dist_ind = np.argsort(np.abs(data - data[loc_max_ind] / scale))
    return [dist_ind[dist_ind <= loc_max_ind][0], dist_ind[dist_ind >= loc_max_ind][0]]


class Slice:
    def __init__(self, start_index=0, end_index=0):
        self.l = start_index
        self.r = end_index

    def set_borders(self, start_index: int, end_index: int) -> None:
        self.l = start_index
        self.r = end_index

    def copy(self, other):
        self.l = other.l
        self.r = other.r

    def check_length(self, len_edge: int) -> bool:
        return self.r - self.l > len_edge

    def check_dist(self, other, dist_edge: int) -> bool:
        return other.l - self.r > dist_edge

    def collide_slices(self, other, dist_edge: int) -> bool:
        if not self.check_dist(other, dist_edge):
            self.r = other.r
            return True
        return False

    def step(self):
        self.r += 1

    def collapse_borders(self):
        self.l = self.r

    def is_null(self) -> bool:
        return self.l == self.r


def process_fragments(data: np.array, mark_data: np.array, length_edge=10, distance_edge=25, scale=1.5, step_out=10) -> np.array:
    proc_slice = Slice(0, 0)
    cur_slice = Slice(0, 1)
    f_fragment = False

    while cur_slice.r < mark_data.shape[0]:
        if mark_data[cur_slice.r] == 1.0:
            if not f_fragment:
                f_fragment = True
        elif f_fragment:
            # print(start_ind, end_ind)
            if scale <= 1:
                if not cur_slice.check_length(length_edge):
                    mark_data[cur_slice.l: cur_slice.r] = 0.0
                elif not proc_slice.collide_slices(cur_slice, distance_edge):
                    mark_data[proc_slice.l: proc_slice.r] = 1.0
                    proc_slice.copy(cur_slice)
            elif scale:
                mark_data[cur_slice.l: cur_slice.r] = 0.0
                if cur_slice.check_length(length_edge):
                    borders = get_borders(data[cur_slice.l: cur_slice.r], scale)
                    # print(boards)
                    borders[0] = max(borders[0] + cur_slice.l - step_out, 0)
                    borders[1] = min(borders[1] + cur_slice.l, mark_data.shape[0])

                    mark_data[borders[0]:borders[1]] = 1.0
            
            f_fragment = False
            cur_slice.collapse_borders()
        elif not f_fragment:
            cur_slice.collapse_borders()
            if proc_slice.is_null():
                proc_slice.copy(cur_slice)
            
        cur_slice.step()

    return mark_data


def focal_loss(y_true, y_pred, alpha=0.1, gamma=2.0):
    bce = K.binary_crossentropy(y_true, y_pred)

    y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())
    p_t = (y_true * y_pred) + ((1 - y_true) * (1 - y_pred))

    alpha_factor = y_true * alpha + ((1 - alpha) * (1 - y_true))
    modulating_factor = K.pow((1 - p_t), gamma)

    # compute the final loss and return
    return K.mean(alpha_factor * modulating_factor * bce, axis=-1)


def dice_bce_loss(y_pred, y_true):
    def dice_loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.math.sigmoid(y_pred)
        numerator = 2 * tf.reduce_sum(y_true * y_pred)
        denominator = tf.reduce_sum(y_true + y_pred)

        return 1 - numerator / denominator

    total_loss = 0.25 * dice_loss(y_pred, y_true) + tf.keras.losses.binary_crossentropy(y_pred, y_true)
    return total_loss


def get_prediction_unet(data: np.array, POINTS_DIM=1024, ckpt_v=0, old=False) -> np.array:
    checkpoint_filepath = f'models/ckpt/checkpoint_{ckpt_v}.weights.h5'

    precision = tf.keras.metrics.Precision()
    recall = tf.keras.metrics.Recall()

    if old:
        model = unet_model_old(POINTS_DIM)
    else:
        model = unet_model(POINTS_DIM)

    model.compile(optimizer='adam', loss=dice_bce_loss,
                  metrics=['acc', precision, recall])
    model.load_weights(checkpoint_filepath)

    l_edge = 0
    step = 256

    prediction_result = np.zeros(data.shape[0])

    while l_edge + POINTS_DIM < data.shape[0]:
        predictions = model.predict(np.array([normalise_series(data[l_edge:l_edge + POINTS_DIM])]), verbose=0)
        for i in range(0, POINTS_DIM):
            prediction_result[l_edge + i] = predictions[0][i]
        l_edge += step

    if l_edge + POINTS_DIM - step != data.shape[0] - 1:
        predictions = model.predict(np.array([normalise_series(data[data.shape[0] - POINTS_DIM:])]), verbose=0)
        for i in range(0, POINTS_DIM):
            prediction_result[data.shape[0] - POINTS_DIM + i] = predictions[0][i]

    # prediction_result = down_to_zero(prediction_result, edge=0.5)
    # prediction_result = process_fragments(data, prediction_result, edge=10, scale=0)

    return prediction_result


def get_prediction_multi_unet(data: np.array, POINTS_DIM=1024, NUM_CLASSES=2, ckpt_v=0) -> np.array:
    checkpoint_filepath = f'models/ckpt/checkpoint_{ckpt_v}_multi.weights.h5'

    precision = tf.keras.metrics.Precision()
    recall = tf.keras.metrics.Recall()
    
    def f1_score(y_true, y_pred):
        p = precision(y_true, y_pred)
        r = recall(y_true, y_pred)
        return 2 * ((p * r) / (p + r + tf.keras.backend.epsilon()))
    
    model = unet_multi_model(POINTS_DIM, NUM_CLASSES)
    model.compile(optimizer='adam', loss=dice_bce_loss,
                  metrics=['acc', precision, recall, f1_score])
    model.load_weights(checkpoint_filepath)
    
    l_edge = 0
    step = 256
    
    prediction_result = np.zeros((NUM_CLASSES, data.shape[0]))
    # print(prediction_result.shape, prediction_result)
    
    while l_edge + POINTS_DIM < data.shape[0]:
        predictions = model.predict(np.array([normalise_series(data[l_edge:l_edge + POINTS_DIM])]), verbose=0)
        # print(predictions.shape)
        for i in range(0, POINTS_DIM):
            prediction_result[0, l_edge + i] = predictions[0, i, 0]
            prediction_result[1, l_edge + i] = predictions[0, i, 1]
        l_edge += step
    
    if l_edge + POINTS_DIM - step != data.shape[0] - 1:
        predictions = model.predict(np.array([normalise_series(data[data.shape[0] - POINTS_DIM:])]), verbose=0)
        for i in range(0, POINTS_DIM):
            prediction_result[0, data.shape[0] - POINTS_DIM + i] = predictions[0, i, 0]
            prediction_result[1, data.shape[0] - POINTS_DIM + i] = predictions[0, i, 1]

    return prediction_result


def unet_multi_model(POINTS_DIM, NUM_CLASSES):
    init_relu = tf.keras.initializers.HeUniform()

    #Входной слой
    inputs = tf.keras.layers.Input(shape=(POINTS_DIM, 1,))
    conv_1 = tf.keras.layers.Conv1D(64, 4, 
                                    activation=tf.keras.layers.LeakyReLU(),
                                    strides=2, padding='same', 
                                    kernel_initializer=init_relu,
                                    use_bias=False)(inputs)
    #Сворачиваем
    conv_1_1 = tf.keras.layers.Conv1D(128, 4, 
                                      activation=tf.keras.layers.LeakyReLU(), 
                                      strides=2,
                                      padding='same', 
                                      kernel_initializer=init_relu,
                                      use_bias=False)(conv_1)
    batch_norm_1 = tf.keras.layers.BatchNormalization()(conv_1_1)

    #2
    conv_2 = tf.keras.layers.Conv1D(256, 4, 
                                    activation=tf.keras.layers.LeakyReLU(), 
                                    strides=2,
                                    padding='same', 
                                    kernel_initializer=init_relu,
                                    use_bias=False)(batch_norm_1)
    batch_norm_2 = tf.keras.layers.BatchNormalization()(conv_2)

    #3
    conv_3 = tf.keras.layers.Conv1D(512, 4, 
                                    activation=tf.keras.layers.LeakyReLU(), 
                                    strides=2,
                                    padding='same', 
                                    kernel_initializer=init_relu,
                                    use_bias=False)(batch_norm_2)
    batch_norm_3 = tf.keras.layers.BatchNormalization()(conv_3)

    #4
    conv_4 = tf.keras.layers.Conv1D(512, 4, 
                                    activation=tf.keras.layers.LeakyReLU(), 
                                    strides=2,
                                    padding='same', 
                                    kernel_initializer=init_relu,
                                    use_bias=False)(batch_norm_3)
    batch_norm_4 = tf.keras.layers.BatchNormalization()(conv_4)

    #Разворачиваем
    #1
    up_1 = tf.keras.layers.Concatenate()([tf.keras.layers.Conv1DTranspose(512, 4, activation='relu', strides=2,
                                                                          padding='same',
                                                                          kernel_initializer=init_relu,
                                                                          use_bias=False)(conv_4), conv_3])
    batch_up_1 = tf.keras.layers.BatchNormalization()(up_1)

    #Добавим Dropout от переобучения
    batch_up_1 = tf.keras.layers.Dropout(0.25)(batch_up_1, training=True)

    #2
    up_2 = tf.keras.layers.Concatenate()([tf.keras.layers.Conv1DTranspose(256, 4, activation='relu', strides=2,
                                                                          padding='same',
                                                                          kernel_initializer=init_relu,
                                                                          use_bias=False)(batch_up_1), conv_2])
    batch_up_2 = tf.keras.layers.BatchNormalization()(up_2)
    batch_up_2 = tf.keras.layers.Dropout(0.25)(batch_up_2, training=True)


    #3
    up_3 = tf.keras.layers.Concatenate()([tf.keras.layers.Conv1DTranspose(128, 4, activation='relu', strides=2,
                                                                          padding='same',
                                                                          kernel_initializer=init_relu,
                                                                          use_bias=False)(batch_up_2), conv_1_1])
    batch_up_3 = tf.keras.layers.BatchNormalization()(up_3)
    batch_up_3 = tf.keras.layers.Dropout(0.25)(batch_up_3, training=True)


    #4
    up_4 = tf.keras.layers.Concatenate()([tf.keras.layers.Conv1DTranspose(64, 4, activation='relu', strides=2,
                                                                          padding='same',
                                                                          kernel_initializer=init_relu,
                                                                          use_bias=False)(batch_up_3), conv_1])
    batch_up_4 = tf.keras.layers.BatchNormalization()(up_4)


    #Выходной слой
    outputs = tf.keras.layers.Conv1DTranspose(NUM_CLASSES, 4, activation='sigmoid', strides=2,
                                                   padding='same',
                                                   kernel_initializer='glorot_normal')(batch_up_4)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def unet_model(POINTS_DIM=1024):

    #Входной слой
    inputs = tf.keras.layers.Input(shape=(POINTS_DIM, 1,))
    conv_1 = tf.keras.layers.Conv1D(64, 4, 
                                    activation=tf.keras.layers.LeakyReLU(),
                                    strides=2, padding='same', 
                                    kernel_initializer='glorot_normal',
                                    use_bias=False)(inputs)
    #Сворачиваем
    conv_1_1 = tf.keras.layers.Conv1D(128, 4, 
                                      activation=tf.keras.layers.LeakyReLU(), 
                                      strides=2,
                                      padding='same', 
                                      kernel_initializer='glorot_normal',
                                      use_bias=False)(conv_1)
    batch_norm_1 = tf.keras.layers.BatchNormalization()(conv_1_1)

    #2
    conv_2 = tf.keras.layers.Conv1D(256, 4, 
                                    activation=tf.keras.layers.LeakyReLU(), 
                                    strides=2,
                                    padding='same', 
                                    kernel_initializer='glorot_normal',
                                    use_bias=False)(batch_norm_1)
    batch_norm_2 = tf.keras.layers.BatchNormalization()(conv_2)

    #3
    conv_3 = tf.keras.layers.Conv1D(512, 4, 
                                    activation=tf.keras.layers.LeakyReLU(), 
                                    strides=2,
                                    padding='same', 
                                    kernel_initializer='glorot_normal',
                                    use_bias=False)(batch_norm_2)
    batch_norm_3 = tf.keras.layers.BatchNormalization()(conv_3)

    #4
    conv_4 = tf.keras.layers.Conv1D(512, 4, 
                                    activation=tf.keras.layers.LeakyReLU(), 
                                    strides=2,
                                    padding='same', 
                                    kernel_initializer='glorot_normal',
                                    use_bias=False)(batch_norm_3)
    batch_norm_4 = tf.keras.layers.BatchNormalization()(conv_4)

    #Разворачиваем
    #1
    up_1 = tf.keras.layers.Concatenate()([tf.keras.layers.Conv1DTranspose(512, 4, activation='relu', strides=2,
                                                                          padding='same',
                                                                          kernel_initializer='glorot_normal',
                                                                          use_bias=False)(conv_4), conv_3])
    batch_up_1 = tf.keras.layers.BatchNormalization()(up_1)

    #Добавим Dropout от переобучения
    batch_up_1 = tf.keras.layers.Dropout(0.25)(batch_up_1, training=True)

    #2
    up_2 = tf.keras.layers.Concatenate()([tf.keras.layers.Conv1DTranspose(256, 4, activation='relu', strides=2,
                                                                          padding='same',
                                                                          kernel_initializer='glorot_normal',
                                                                          use_bias=False)(batch_up_1), conv_2])
    batch_up_2 = tf.keras.layers.BatchNormalization()(up_2)
    batch_up_2 = tf.keras.layers.Dropout(0.25)(batch_up_2, training=True)




    #3
    up_3 = tf.keras.layers.Concatenate()([tf.keras.layers.Conv1DTranspose(128, 4, activation='relu', strides=2,
                                                                          padding='same',
                                                                          kernel_initializer='glorot_normal',
                                                                          use_bias=False)(batch_up_2), conv_1_1])
    batch_up_3 = tf.keras.layers.BatchNormalization()(up_3)
    batch_up_3 = tf.keras.layers.Dropout(0.25)(batch_up_3, training=True)




    #4
    up_4 = tf.keras.layers.Concatenate()([tf.keras.layers.Conv1DTranspose(64, 4, activation='relu', strides=2,
                                                                          padding='same',
                                                                          kernel_initializer='glorot_normal',
                                                                          use_bias=False)(batch_up_3), conv_1])
    batch_up_4 = tf.keras.layers.BatchNormalization()(up_4)


    #Выходной слой
    max_pool = tf.keras.layers.MaxPooling1D(pool_size=2)(batch_up_4)
    flat = tf.keras.layers.Flatten()(max_pool)
    flat = tf.keras.layers.Dropout(0.2)(flat, training=True)
    outputs = tf.keras.layers.Dense(POINTS_DIM, activation="sigmoid")(flat)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def unet_model_old(POINTS_DIM=1024):
    # Входной слой
    inputs = tf.keras.layers.Input(shape=(POINTS_DIM, 1,))
    conv_1 = tf.keras.layers.Conv1D(64, 4,
                                    activation=tf.keras.layers.LeakyReLU(),
                                    strides=2, padding='same',
                                    kernel_initializer='glorot_normal',
                                    use_bias=False)(inputs)
    # Сворачиваем
    conv_1_1 = tf.keras.layers.Conv1D(128, 4,
                                      activation=tf.keras.layers.LeakyReLU(),
                                      strides=2,
                                      padding='same',
                                      kernel_initializer='glorot_normal',
                                      use_bias=False)(conv_1)
    batch_norm_1 = tf.keras.layers.BatchNormalization()(conv_1_1)

    # 2
    conv_2 = tf.keras.layers.Conv1D(256, 4,
                                    activation=tf.keras.layers.LeakyReLU(),
                                    strides=2,
                                    padding='same',
                                    kernel_initializer='glorot_normal',
                                    use_bias=False)(batch_norm_1)
    batch_norm_2 = tf.keras.layers.BatchNormalization()(conv_2)

    # 3
    conv_3 = tf.keras.layers.Conv1D(512, 4,
                                    activation=tf.keras.layers.LeakyReLU(),
                                    strides=2,
                                    padding='same',
                                    kernel_initializer='glorot_normal',
                                    use_bias=False)(batch_norm_2)
    batch_norm_3 = tf.keras.layers.BatchNormalization()(conv_3)

    # 4
    conv_4 = tf.keras.layers.Conv1D(512, 4,
                                    activation=tf.keras.layers.LeakyReLU(),
                                    strides=2,
                                    padding='same',
                                    kernel_initializer='glorot_normal',
                                    use_bias=False)(batch_norm_3)
    batch_norm_4 = tf.keras.layers.BatchNormalization()(conv_4)

    # 5
    conv_5 = tf.keras.layers.Conv1D(512, 4,
                                    activation=tf.keras.layers.LeakyReLU(),
                                    strides=2,
                                    padding='same',
                                    kernel_initializer='glorot_normal',
                                    use_bias=False)(batch_norm_4)
    batch_norm_5 = tf.keras.layers.BatchNormalization()(conv_5)

    # 6
    conv_6 = tf.keras.layers.Conv1D(512, 4,
                                    activation=tf.keras.layers.LeakyReLU(),
                                    strides=2,
                                    padding='same',
                                    kernel_initializer='glorot_normal',
                                    use_bias=False)(batch_norm_5)

    # Разворачиваем
    # 1
    up_1 = tf.keras.layers.Concatenate()([tf.keras.layers.Conv1DTranspose(512, 4, activation='relu', strides=2,
                                                                          padding='same',
                                                                          kernel_initializer='glorot_normal',
                                                                          use_bias=False)(conv_6), conv_5])
    batch_up_1 = tf.keras.layers.BatchNormalization()(up_1)

    # Добавим Dropout от переобучения
    batch_up_1 = tf.keras.layers.Dropout(0.25)(batch_up_1)

    # 2
    up_2 = tf.keras.layers.Concatenate()([tf.keras.layers.Conv1DTranspose(512, 4, activation='relu', strides=2,
                                                                          padding='same',
                                                                          kernel_initializer='glorot_normal',
                                                                          use_bias=False)(batch_up_1), conv_4])
    batch_up_2 = tf.keras.layers.BatchNormalization()(up_2)
    batch_up_2 = tf.keras.layers.Dropout(0.25)(batch_up_2)

    # 3
    up_3 = tf.keras.layers.Concatenate()([tf.keras.layers.Conv1DTranspose(512, 4, activation='relu', strides=2,
                                                                          padding='same',
                                                                          kernel_initializer='glorot_normal',
                                                                          use_bias=False)(batch_up_2), conv_3])
    batch_up_3 = tf.keras.layers.BatchNormalization()(up_3)
    batch_up_3 = tf.keras.layers.Dropout(0.25)(batch_up_3)

    # 4
    up_4 = tf.keras.layers.Concatenate()([tf.keras.layers.Conv1DTranspose(256, 4, activation='relu', strides=2,
                                                                          padding='same',
                                                                          kernel_initializer='glorot_normal',
                                                                          use_bias=False)(batch_up_3), conv_2])
    batch_up_4 = tf.keras.layers.BatchNormalization()(up_4)

    # 5
    up_5 = tf.keras.layers.Concatenate()([tf.keras.layers.Conv1DTranspose(128, 4, activation='relu', strides=2,
                                                                          padding='same',
                                                                          kernel_initializer='glorot_normal',
                                                                          use_bias=False)(batch_up_4), conv_1_1])
    batch_up_5 = tf.keras.layers.BatchNormalization()(up_5)

    # 6
    up_6 = tf.keras.layers.Concatenate()([tf.keras.layers.Conv1DTranspose(64, 4, activation='relu', strides=2,
                                                                          padding='same',
                                                                          kernel_initializer='glorot_normal',
                                                                          use_bias=False)(batch_up_5), conv_1])
    batch_up_6 = tf.keras.layers.BatchNormalization()(up_6)

    # Выходной слой
    max_pool = tf.keras.layers.MaxPooling1D(pool_size=2)(batch_up_6)
    flat = tf.keras.layers.Flatten()(max_pool)
    flat = tf.keras.layers.Dropout(0.1)(flat, training=True)
    outputs = tf.keras.layers.Dense(POINTS_DIM, activation="sigmoid")(flat)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model
