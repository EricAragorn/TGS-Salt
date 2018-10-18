import cv2 as cv
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class Config:
    TRAINING_SET_RATIO = 0.85

    # Data parameters
    img_size = 101
    batch_size = 32

    # Model parameters
    UNet_layers = 4

    # Initial_Conv_Kernel_Size = (7, 7)
    # Initial_Conv_Filters = 64
    Conv_Kernel_Size = (3, 3)
    Conv_Kernel_Initializer = tf.initializers.he_normal()
    # Conv_Kernel_Initializer = tf.initializers.constant(0.1)
    Conv_Filter_Size = [16, 32, 64, 128, 256]
    # Conv_Filter_Size = [8, 16, 32, 64, 128]
    Conv_Bottleneck_Size = [int(i / 2) for i in Conv_Filter_Size]
    Stacked_ResBlock_Depth = [2, 2, 2, 2, 2]
    Dropout_Ratio = 0.2

    Max_Epoch = 50
    Max_steps = 150
    initial_lr = 0.0005
    lr_decay_rate = 0.95
    lr_decay_after = 30
    min_lr = 0.0001

    Up_Block_Padding = ['valid', 'same', 'valid', 'same']


class Dataset:
    def __init__(self):
        # TODO Target_data must be converted to mask

        # # use `itertuples` instead of `iterrows` for performance reason; row[1] = id, row[2] = z
        # data = [self.to_dict(row[1], row[2]) for row in pd.read_csv("depths.csv").itertuples()]
        #
        # # remove data entries with no image/mask
        # data = [img for img in data if img["mask"] is not None]

        ######################################################################################################
        # code from https://www.kaggle.com/shaojiaxin/u-net-with-simple-resnet-blocks-v2-new-loss
        # Special thanks to the author Jack (Jiaxin) Shao
        train_df = pd.read_csv("train.csv", index_col="id", usecols=[0])
        depth_df = pd.read_csv("depths.csv", index_col="id")
        train_df = train_df.join(depth_df)

        train_df["images"] = [
            np.array(cv.imread(f"train/images/{idx}.png", flags=cv.IMREAD_GRAYSCALE), dtype=np.float32) / 255
            for idx in train_df.index]
        train_df["masks"] = [
            np.array(cv.imread(f"train/masks/{idx}.png", flags=cv.IMREAD_GRAYSCALE), dtype=np.float32) / 255
            for idx in train_df.index]

        train_df["coverage"] = train_df["masks"].map(np.sum) / pow(Config.img_size, 2)

        def cov_to_class(val):
            for i in range(0, 11):
                if val * 10 <= i:
                    return i

        train_df["coverage_class"] = train_df.coverage.map(cov_to_class)

        x_train, x_valid, y_train, y_valid, cov_train, cov_valid, depth_train, depth_valid = train_test_split(
            np.array(train_df["images"].tolist()).reshape(-1, Config.img_size, Config.img_size, 1),
            np.array(train_df["masks"].tolist()).reshape(-1, Config.img_size, Config.img_size, 1),
            train_df.coverage.values,
            train_df["z"].values,
            test_size=0.2, stratify=train_df["coverage_class"], random_state=1234)

        x_train = np.append(x_train, [np.fliplr(x) for x in x_train], axis=0)
        y_train = np.append(y_train, [np.fliplr(x) for x in y_train], axis=0)
        # f, ax = plt.subplots(2, 2)
        # plt.imshow(train_df["images"][0])
        # ax[0, 0].imshow(np.reshape(train_df["images"][0], newshape=(101, 101)), cmap="gray")
        # ax[0, 1].imshow(np.reshape(y_train[0], newshape=(101, 101)), cmap="gray")
        # plt.show()
        #######################################################################################################

        self.train = (x_train, y_train)
        self.valid = (x_valid, y_valid)
    # self.batch_base_ptr = 0

    # def next_training_batch(self, size=32):
    #     if self.batch_base_ptr + size >= len(self.training):
    #         batch = self.training[self.batch_base_ptr:len(self.training)]
    #         self.batch_base_ptr = 0
    #     else:
    #         batch = self.training[self.batch_base_ptr: self.batch_base_ptr + size]
    #         self.batch_base_ptr = self.batch_base_ptr + size
    #     return self.valid_output(batch)
    #
    # def get_validation(self):
    #     return self.valid_output(self.validation)

    # @staticmethod
    # def to_dict(id, z):
    #     return {
    #         "image": np.array(load_img(f"./train/masks/{id}.png", grayscale=True)) / 255,
    #         "mask": np.array(load_img(f"./train/images/{id}.png", grayscale=True)) / 255,
    #         "z": z,
    #         "id": id
    #     }
    #
    # @staticmethod
    # def valid_output(data):
    #     return {
    #         "image": [e['image'].reshape((101, 101, 1)) for e in data],
    #         "z": [e['z'] for e in data],
    #         "mask": [e['mask'].reshape((101, 101, 1)) for e in data]
    #     }


class TGSModel:
    """
    Initialize the TGS Salt Detection model

    Work based on the following papers:
        Deep Residual Learning for Image Recognition by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.
        Identity Mappings in Deep Residual Networks by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.
    """

    # TODO Fix arithmetic exceptions in upsample resblocks
    def __init__(self, optimizer='Adam'):
        TGS_db = Dataset()

        self.train_x, self.train_y = TGS_db.train
        self.valid_x, self.valid_y = TGS_db.valid
        # self.test_x, self.test_y = tf.placeholder(dtype=np.float32, shape=(None, 101, 101, 1)), \
        #                            tf.placeholder(dtype=np.float32, shape=(None, 101, 101, 1))

        # construct tf dataset
        train_set = tf.data.Dataset.from_tensor_slices((self.train_x, self.train_y)) \
            .batch(Config.batch_size)
        valid_set = tf.data.Dataset.from_tensor_slices((self.valid_x, self.valid_y)) \
            .batch(Config.batch_size)
        # test_set = tf.data.Dataset.from_tensor_slices((self.test_x, self.test_y))

        iter = tf.data.Iterator.from_structure(train_set.output_types, train_set.output_shapes)
        self.train_init_op = iter.make_initializer(train_set)
        self.valid_init_op = iter.make_initializer(valid_set)
        # self.test_init_op = iter.make_initializer(test_set)
        img, mask = iter.get_next()

        self.name = "TGS_UNet_w_ResBlocks_v1"

        with tf.name_scope("TGS_input"):
            self.input = img
        with tf.name_scope("TGS_target"):
            self.target = mask

        """
        Network architechture
        L1 input---------------------------------->mask -> final mask
        L2    down1------------------------------>up1
        L3       down2--------------------->up2
        L4          down3 ------------->up3
        L5              down4------->up4
        L6                  features
        """

        prev = self.input
        down_block_output = []
        up_block_output = []
        # downsampling layers
        for down_block_id in range(Config.UNet_layers):
            with tf.variable_scope(f"{self.name}/down{down_block_id}"):
                conv = tf.layers.conv2d(inputs=prev,
                                        kernel_size=(3, 3),
                                        kernel_initializer=Config.Conv_Kernel_Initializer,
                                        filters=Config.Conv_Filter_Size[down_block_id],
                                        padding='same')
                # resblocks = tf.nn.leaky_relu(conv)
                resblocks = TGSModel.stacked_res_blocks(inputs=conv,
                                                        kernel_size=Config.Conv_Kernel_Size,
                                                        filters=Config.Conv_Filter_Size[down_block_id],
                                                        bottleneck_filters=Config.Conv_Bottleneck_Size[down_block_id],
                                                        count=Config.Stacked_ResBlock_Depth[down_block_id])
                pooling = tf.layers.max_pooling2d(inputs=resblocks,
                                                  pool_size=(2, 2),
                                                  strides=2)
                prev = pooling
                # dropout = tf.layers.dropout(inputs=pooling, rate=Config.Dropout_Ratio)
                down_block_output.append(resblocks)

        # middle layer
        with tf.variable_scope(f"{self.name}/mid"):
            conv = tf.layers.conv2d(inputs=prev,
                                    kernel_size=(3, 3),
                                    kernel_initializer=Config.Conv_Kernel_Initializer,
                                    filters=Config.Conv_Filter_Size[Config.UNet_layers],
                                    padding="same")
            self._test_ = conv
            # self.middle_out = tf.nn.leaky_relu(conv)
            self.middle_out = TGSModel.stacked_res_blocks(inputs=conv,
                                                          kernel_size=Config.Conv_Kernel_Size,
                                                          filters=Config.Conv_Filter_Size[Config.UNet_layers],
                                                          count=Config.Stacked_ResBlock_Depth[Config.UNet_layers])
        prev = self.middle_out
        #
        # upsampling layers
        for up_block_id in range(Config.UNet_layers - 1, -1, -1):
            with tf.variable_scope(f"{self.name}/up{up_block_id}"):
                # dropout = tf.layers.dropout(up_block_output[0], rate=Config.Dropout_Ratio)
                conv1 = tf.layers.conv2d_transpose(inputs=prev,
                                                   filters=Config.Conv_Filter_Size[up_block_id],
                                                   kernel_size=(3, 3),
                                                   kernel_initializer=Config.Conv_Kernel_Initializer,
                                                   padding=Config.Up_Block_Padding[up_block_id],
                                                   strides=(2, 2),
                                                   name="conv1")
                concat = tf.concat([down_block_output[up_block_id], conv1], axis=-1, name="concat")
                conv2 = tf.layers.conv2d(inputs=concat,
                                         filters=Config.Conv_Filter_Size[up_block_id],
                                         kernel_size=(3, 3),
                                         kernel_initializer=Config.Conv_Kernel_Initializer,
                                         padding="same",
                                         name="conv2")
                # resblocks = tf.nn.leaky_relu(conv2)
                resblocks = TGSModel.stacked_res_blocks(inputs=conv2,
                                                        kernel_size=Config.Conv_Kernel_Size,
                                                        filters=Config.Conv_Filter_Size[up_block_id],
                                                        bottleneck_filters=Config.Conv_Bottleneck_Size[up_block_id],
                                                        count=Config.Stacked_ResBlock_Depth[up_block_id])
                prev = resblocks
                up_block_output.insert(0, resblocks)
        # mask
        gen_mask_noActivation = tf.layers.conv2d(inputs=up_block_output[0],
                                                 filters=1,
                                                 kernel_size=(1, 1),
                                                 kernel_initializer=Config.Conv_Kernel_Initializer)
        # self.gen_mask = tf.sigmoid(gen_mask_noActivation)

        # compute loss
        self.loss = tf.losses.sigmoid_cross_entropy(self.target, gen_mask_noActivation)

        self.lr = tf.placeholder(dtype=tf.float32, shape=[])
        if optimizer == 'Adam':
            _optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        elif optimizer == 'SGD':
            _optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        else:
            raise ValueError("Unsupported optimizer")
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            gradients, values = zip(*_optimizer.compute_gradients(self.loss))
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5)
            self.train_op = _optimizer.apply_gradients(zip(clipped_gradients, values))

        self.prediction = tf.round(tf.sigmoid(gen_mask_noActivation))
        inter = tf.reduce_sum(tf.multiply(self.prediction, self.target))
        union = tf.reduce_sum(tf.subtract(tf.add(self.prediction, self.target), tf.multiply(self.prediction, self.target)))
        self.iou_vector = inter / union

    @staticmethod
    def stacked_res_blocks(inputs, kernel_size, filters, count, bottleneck_filters=None, type="resblock"):
        if count < 1:
            raise ValueError("The number of stacked residual blocks should be positive")

        if type == "resblock":
            # Original ResBlock
            last_block = inputs
            for i in range(count - 1):
                last_block = TGSModel.resBlock(inputs=last_block,
                                               kernel_size=kernel_size,
                                               filters=filters,
                                               block_id=i)

            last_block = TGSModel.resBlock(inputs=last_block,
                                           kernel_size=kernel_size,
                                           filters=filters,
                                           block_id=count - 1,
                                           activation=True)

            # resB1 = TGSModel.resBlock(inputs=inputs,
            #                           kernel_size=kernel_size,
            #                           filters=filters,
            #                           block_id=0)
            # resB2 = TGSModel.resBlock(inputs=resB1,
            #                           kernel_size=kernel_size,
            #                           filters=filters,
            #                           block_id=1,
            #                           activation=True)
        else:
            # Bottleneck ResBlock
            if bottleneck_filters is None:
                raise ValueError("Bottleneck filter size must be specified for bottleneck resblocks")
            last_block = TGSModel.bottleneck_block(
                                                   inputs=inputs,
                                                   kernel_size=kernel_size,
                                                   filters=filters,
                                                   bottleneck_filters=bottleneck_filters,
                                                   block_id=0,
                                                   shortcut=False)

            for i in range(count - 1):
                last_block = TGSModel.bottleneck_block(inputs=last_block,
                                                       kernel_size=kernel_size,
                                                       filters=filters,
                                                       bottleneck_filters=bottleneck_filters,
                                                       block_id=i + 1)

        return last_block

    # ResBlock_v2
    @staticmethod
    def resBlock(inputs, kernel_size, filters, block_id, strides=(1, 1), activation=False):
        with tf.variable_scope(f"ResBlock{block_id}"):
            bn1 = tf.layers.batch_normalization(inputs=inputs, name="bn1")
            relu1 = tf.nn.leaky_relu(bn1, name="relu1")
            conv1 = tf.layers.conv2d(inputs=relu1,
                                     filters=filters,
                                     kernel_size=kernel_size,
                                     kernel_initializer=Config.Conv_Kernel_Initializer,
                                     strides=strides,
                                     padding="same",
                                     name="conv1")
            bn2 = tf.layers.batch_normalization(inputs=conv1, name="bn2")
            relu2 = tf.nn.leaky_relu(bn2, name="relu2")
            conv2 = tf.layers.conv2d(inputs=relu2,
                                     filters=filters,
                                     kernel_size=kernel_size,
                                     kernel_initializer=Config.Conv_Kernel_Initializer,
                                     strides=strides,
                                     padding="same",
                                     name="conv2")
            output = conv2 + inputs
            if activation:
                output = tf.layers.batch_normalization(output, name="output_bn")
                output = tf.nn.leaky_relu(output, name="output_activation")
            return output


    """
    Implementation of ResNet bottleneck block

    For more details, please refer to:
    https://arxiv.org/pdf/1603.05027.pdf
    Identity Mappings in Deep Residual Networks by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.
    """

    @staticmethod
    def bottleneck_block(self, inputs, kernel_size, filters, bottleneck_filters, block_id, strides=1,
                         shortcut=True):
        BNConv_filters1 = BNConv_filters2 = bottleneck_filters

        with tf.variable_scope(f"BottleNeckResBlock{block_id}"):
            bn1 = tf.layers.batch_normalization(inputs=inputs,
                                                name="bn1")
            relu1 = tf.nn.relu(bn1)
            conv1 = tf.layers.conv2d(inputs=relu1,
                                     kernel_size=(1, 1),
                                     filters=BNConv_filters1,
                                     kernel_initializer=Config.Conv_Kernel_Initializer,
                                     name="conv1")

            bn2 = tf.layers.batch_normalization(inputs=conv1,
                                                name="bn2")
            relu2 = tf.nn.relu(bn2)
            conv2 = tf.layers.conv2d(inputs=relu2,
                                     kernel_size=kernel_size,
                                     filters=BNConv_filters2,
                                     strides=strides,
                                     padding="same",
                                     kernel_initializer=Config.Conv_Kernel_Initializer,
                                     name="conv2")

            bn3 = tf.layers.batch_normalization(inputs=conv2,
                                                name="bn3")
            relu3 = tf.nn.relu(bn3)
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
    m = TGSModel()
    # saver = tf.train.Saver()
    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter("./log", sess.graph)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        lr = Config.initial_lr
        print("Start Training...")
        # sess.run([m.img, m.mask])
        for epoch in range(Config.Max_Epoch):
            print("Epoch %d/%d:" % (epoch + 1, Config.Max_Epoch))
            # Apply learning rate decay
            if epoch > Config.lr_decay_after and lr > Config.min_lr:
                lr *= Config.lr_decay_rate

            epoch_train_loss = []
            epoch_train_iou = []

            sess.run(m.train_init_op)
            for step in range(Config.Max_steps):
                _, train_loss, train_iou = sess.run([m.train_op, m.loss, m.iou_vector], feed_dict={m.lr: lr})
                epoch_train_loss.append(train_loss)
                epoch_train_iou.append(train_iou)

            epoch_val_loss = []
            epoch_val_iou = []
            sess.run(m.valid_init_op)
            for i in range(10):
                loss, iou = sess.run([m.loss, m.iou_vector], feed_dict={m.lr: lr})
                epoch_val_loss.append(loss)
                epoch_val_iou.append(iou)
            print("\t training loss: %s, train_iou: %s, validation loss: %s, validation_iou: %s"
                  % (np.mean(epoch_train_loss), np.mean(epoch_train_iou), np.mean(epoch_val_loss), np.mean(epoch_val_iou)))
            # if (epoch + 1) % 5000:
            #     saver.save(sess, 'UNet2-ResNet34', global_step=epoch+1)

        def gen_visualization(dir):
            input, target, mask, middle_out, _test_ = sess.run([m.input, m.target, m.prediction, m.middle_out, m._test_])

            for id in range(input.shape[0]):
                input_img, mask_img, target_img = np.reshape(input[id], newshape=(101, 101)) * 255, \
                                                  np.reshape(mask[id], newshape=(101, 101)) * 255, \
                                                  np.reshape(target[id], newshape=(101, 101)) * 255
                f, ax = plt.subplots(1, 3)
                ax[0].imshow(input_img, cmap='gray')
                ax[1].imshow(mask_img, cmap='gray')
                ax[2].imshow(target_img, cmap='gray')
                plt.savefig("%s/%d.png" % (dir, id))
                plt.close()

                fig, ax = plt.subplots(int(middle_out.shape[3] / 20) + 1, 20)
                for filter in range(middle_out.shape[3]):
                    ax[int(filter / 20), int(filter % 20)].imshow(np.reshape(middle_out[id, :, :, int(filter)],
                                                                             newshape=(middle_out.shape[1],
                                                                                       middle_out.shape[2])) * 255)
                    ax[int(filter / 20), int(filter % 20)].axis('off')
                plt.savefig("%s/%d_mid.png" % (dir, id))
                plt.close()

                fig, ax = plt.subplots(int(_test_.shape[3] / 20) + 1, 20)
                for filter in range(_test_.shape[3]):
                    ax[int(filter / 20), int(filter % 20)].imshow(np.reshape(_test_[id, :, :, int(filter)],
                                                                             newshape=(_test_.shape[1],
                                                                                       _test_.shape[2])) * 255)
                    ax[int(filter / 20), int(filter % 20)].axis('off')
                plt.savefig("%s/%d_test.png" % (dir, id))
                plt.close()

        sess.run(m.train_init_op)
        gen_visualization("train_output")

        sess.run(m.valid_init_op)
        gen_visualization("valid_output")


def test():
    x = tf.convert_to_tensor(np.array([1, 0], dtype=np.float32))
    y = tf.convert_to_tensor(np.array([0, 0], dtype=np.float32))
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=x)
    iou, initializer = tf.metrics.mean_iou(x, y, num_classes=2)
    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(initializer)
        print(sess.run(loss))
        print(sess.run(iou))


def testResBlock():
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = tf.convert_to_tensor(np.reshape(np.array(x_train), newshape=(-1, 28, 28, 1)), dtype=tf.float32)[0:500]
    y_train = tf.convert_to_tensor(np.reshape(np.array(y_train), newshape=(-1, 1)), dtype=tf.int32)[0:500]
    y_train = tf.one_hot(y_train, depth=10)
    print(y_train.shape)
    # conv1 = tf.layers.conv2d(inputs=x_train,
    #                          filters=32,
    #                          kernel_size=(5, 5))
    resB = TGSModel.stacked_res_blocks(inputs=x_train,
                                       kernel_size=(3, 3),
                                       filters=32,
                                       count=8)
    flatten = tf.layers.flatten(resB)
    pred = tf.layers.dense(inputs=flatten,
                           units=10)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_train, logits=pred))
    train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(1000):
            sess.run(train_op)
            print("Step: %d, Loss %f" % (i, sess.run(loss)))


if __name__ == "__main__":
    main()
