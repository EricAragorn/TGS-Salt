import gc

import cv2 as cv
import pandas as pd
import tensorflow as tf
from sklearn.utils import shuffle


class Config:
    TRAINING_SET_RATIO = 0.75

    # Model parameters
    UNet_layers = 1

    Initial_Conv_Kernel_Size = (7, 7)
    Initial_Conv_Filters = 64
    Conv_Kernel_Size = (3, 3)
    Conv_Kernel_Initializer = tf.initializers.truncated_normal(stddev=0.5)
    Conv_Filter_Size = [1, 16, 32, 64, 128]
    Conv_Bottleneck_Size = [int(i / 4) for i in Conv_Filter_Size]
    Stacked_ResBlock_Depth = [2, 2, 2, 2, 2]
    Dropout_Ratio = 0.2

    Up_Block_Padding = ['valid', 'same', 'valid', 'same']


class Batcher:
    def __init__(self, data):
        self.data = data
        self.init_ptr = 0

    def next_batch(self, size):
        if self.init_ptr + size < self.data.shape[0]:
            batch = self.data[self.init_ptr:self.init_ptr + size - 1, :, :]


class Dataset:
    def __init__(self):
        # TODO Target_data must be converted to mask

        # use `itertuples` instead of `iterrows` for performance reason; row[1] = id, row[2] = z
        data = [self.to_dict(row[1], row[2]) for row in pd.read_csv("depths.csv").itertuples()]

        # remove data entries with no image/mask
        data = [img for img in data if img["mask"] is not None]

        self.num = len(data)

        shuffle(data)

        self.training = data[:int(Config.TRAINING_SET_RATIO * self.num)]
        self.validation = data[int(Config.TRAINING_SET_RATIO * self.num) + 1:]
        self.batch_base_ptr = 0

    # TODO Training batch returns empty after a few steps
    def next_training_batch(self, size=32):
        if self.batch_base_ptr + size >= len(self.training):
            batch = self.training[self.batch_base_ptr:len(self.training)]
            self.batch_base_ptr = 0
        else:
            batch = self.training[self.batch_base_ptr: self.batch_base_ptr + size]
            self.batch_base_ptr = self.batch_base_ptr + size
        return self.output(batch)

    def get_validation(self):
        return self.output(self.validation)

    @staticmethod
    def to_dict(id, z):
        return {
            "image": cv.imread(f"train/masks/{id}.png", cv.IMREAD_GRAYSCALE),
            "mask": cv.imread(f"train/images/{id}.png", cv.IMREAD_GRAYSCALE),
            "z": z,
            "id": id
        }

    @staticmethod
    def output(data):
        return {
            "image": [e['image'].reshape((101, 101, 1)) for e in data],
            "z": [e['z'] for e in data],
            "mask": [e['mask'].reshape((101, 101, 1)) for e in data]
        }


class TGSModel:
    """
    Initialize the TGS Salt Detection model

    Work based on the following papers:
        Deep Residual Learning for Image Recognition by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.
        Identity Mappings in Deep Residual Networks by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.
    """

    def __init__(self, optimizer='SGD'):
        self.name = "TGS_UNet_w_ResBlocks_v1"
        with tf.name_scope("TGS_input"):
            self.input = tf.placeholder(dtype=tf.float32,
                                        shape=(None, 101, 101, 1))
        with tf.name_scope("TGS_target"):
            self.target = tf.placeholder(dtype=tf.float32,
                                         shape=(None, 101, 101, 1))

        """
        Network architechture
        L1 input---------------------------------->mask -> final mask
        L2    down1------------------------------>up1
        L3       down2--------------------->up2
        L4          down3 ------------->up3
        L5              down4------->up4
        L6                  features
        """

        down_block_output = [self.input]
        # downsampling layers
        for down_block_id in range(1, Config.UNet_layers + 1):
            with tf.variable_scope(f"{self.name}/down{down_block_id}"):
                conv = tf.layers.conv2d(inputs=down_block_output[down_block_id - 1],
                                        kernel_size=(3, 3),
                                        filters=Config.Conv_Filter_Size[down_block_id],
                                        padding='same')
                resblocks = TGSModel.stacked_res_blocks(inputs=conv,
                                                        kernel_size=Config.Conv_Kernel_Size,
                                                        filters=Config.Conv_Filter_Size[down_block_id],
                                                        bottleneck_filters=Config.Conv_Bottleneck_Size[down_block_id],
                                                        count=Config.Stacked_ResBlock_Depth[down_block_id])
                pooling = tf.layers.max_pooling2d(inputs=resblocks,
                                                  pool_size=(2, 2),
                                                  strides=2)
                dropout = tf.layers.dropout(inputs=pooling, rate=Config.Dropout_Ratio)
                down_block_output.append(dropout)

        # middle layer
        up_features = down_block_output[Config.UNet_layers]
        #
        # upsampling layers
        # for up_block_id in range(Config.UNet_layers - 1, -1, -1):
        #     with tf.variable_scope(f"{self.name}/up{up_block_id}"):
        #         recovered = tf.layers.conv2d_transpose(inputs=up_features,
        #                                                filters=Config.Conv_Filter_Size[up_block_id],
        #                                                kernel_size=(3, 3),
        #                                                padding=Config.Up_Block_Padding[up_block_id],
        #                                                strides=2)
        #         # print(up_block_id)
        #         # print(down_block_output[up_block_id].shape)
        #         # print(recovered.shape)
        #         # print("\n")
        #         concat = tf.concat([down_block_output[up_block_id], recovered], axis=-1)
        #         concat = tf.layers.conv2d(inputs=concat,
        #                                   filters=Config.Conv_Filter_Size[up_block_id],
        #                                   kernel_size=(1, 1),
        #                                   strides=1)
        #         up_features = TGSModel.stacked_res_blocks(inputs=concat,
        #                                                   kernel_size=Config.Conv_Kernel_Size,
        #                                                   filters=Config.Conv_Filter_Size[up_block_id],
        #                                                   bottleneck_filters=Config.Conv_Bottleneck_Size[up_block_id],
        #                                                   count=Config.Stacked_ResBlock_Depth[up_block_id])
        flattened = tf.layers.flatten(up_features)
        up_features = tf.layers.dense(inputs=flattened, units=10201, activation=tf.tanh)
        # mask
        target_mask = tf.layers.flatten(self.target)
        flattened = tf.layers.flatten(up_features)
        gen_mask = tf.sigmoid(flattened)

        # compute loss
        self.loss = -tf.losses.sigmoid_cross_entropy(target_mask, gen_mask)

        self.lr = tf.placeholder(dtype=tf.float32, shape=[])
        if optimizer == 'Adam':
            _optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        elif optimizer == 'SGD':
            _optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        else:
            raise ValueError("Unsupported optimizer")
        self.train_op = _optimizer.minimize(self.loss)

    @staticmethod
    def stacked_res_blocks(inputs, kernel_size, filters, bottleneck_filters, count):
        if count < 1:
            raise ValueError("The number of stacked residual blocks should be positive")

        last_block = inputs

        for i in range(count):
            with tf.variable_scope(f"Block{i}"):
                last_block = TGSModel.bottleneck_block(inputs=last_block,
                                                       kernel_size=kernel_size,
                                                       filters=filters,
                                                       bottleneck_filters=bottleneck_filters,
                                                       block_id=i + 1)

        return last_block

    """
    Implementation of ResNet bottleneck block

    For more details, please refer to:
    https://arxiv.org/pdf/1603.05027.pdf
    Identity Mappings in Deep Residual Networks by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.
    """

    # TODO Urgent! Error with tensor sizes causes allocation failures
    @staticmethod
    def bottleneck_block(inputs, kernel_size, filters, bottleneck_filters, block_id, strides=1,
                         shortcut=True):
        filter1 = filter2 = bottleneck_filters

        with tf.name_scope("Block{}".format(block_id)):
            bn1 = tf.layers.batch_normalization(inputs=inputs,
                                                name="bn1")
            relu1 = tf.nn.relu(bn1)
            conv1 = tf.layers.conv2d(inputs=relu1,
                                     kernel_size=(1, 1),
                                     filters=filter1,
                                     kernel_initializer=Config.Conv_Kernel_Initializer,
                                     name="conv1")

            bn2 = tf.layers.batch_normalization(inputs=conv1,
                                                name="bn2")
            relu2 = tf.nn.relu(bn2)
            conv2 = tf.layers.conv2d(inputs=relu2,
                                     kernel_size=kernel_size,
                                     filters=filter2,
                                     strides=strides,
                                     padding="same",
                                     kernel_initializer=Config.Conv_Kernel_Initializer,
                                     name="conv2")

            bn3 = tf.layers.batch_normalization(inputs=conv2,
                                                name="bn3")
            relu3 = tf.nn.relu(bn3)
            print(relu3.shape)
            conv3 = tf.layers.conv2d(inputs=relu3,
                                     kernel_size=(1, 1),
                                     filters=filters,
                                     kernel_initializer=Config.Conv_Kernel_Initializer,
                                     name="conv3")

            if shortcut:
                ret = tf.add(inputs, conv3)
            else:
                ret = conv3
            return ret


def main():
    TGS_dataset = Dataset()
    gc.enable()

    m = TGSModel(optimizer='Adam')
    saver = tf.train.Saver()
    val_set = TGS_dataset.get_validation()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("Start Training...")
        for step in range(2000):
            if step % 10 == 0:
                loss = sess.run(m.loss, feed_dict={m.input: val_set['image'], m.target: val_set['mask'], m.lr: 0.001})
                print("Step %d, Loss: %f" % (step, loss))
            batch = TGS_dataset.next_training_batch()
            sess.run(m.train_op, feed_dict={m.input: batch['image'], m.target: batch['mask'], m.lr: 0.001},
                     options=tf.RunOptions(report_tensor_allocations_upon_oom=True))


def test_dataset():
    TGS_dataset = Dataset()
    training_batch = TGS_dataset.next_training_batch()


if __name__ == "__main__":
    main()
