import datetime
import tensorflow as tf
from tensorflow import keras
from models import mlp, lenet, alexnet, restnet, googlenet
from datasets.dataset import DataLoader


class Neural(object):
    def __init__(self, model_type='mlp', batch_size=32, epochs=8, data_path='./'):
        self.is_reshape = True
        self.model_type = model_type
        self.model = self.get_model()
        self.batch_size = batch_size
        self.start_epoch = 0
        self.epochs = epochs
        self.is_train = True
        self.split_factor = 0.1

        self.data_path = data_path
        self.train_dataset = DataLoader(data_path=self.data_path, is_train=True, is_reshape=self.is_reshape,
                                        split_factor=self.split_factor,
                                        batch_size=self.batch_size)
        self.val_dataset = DataLoader(data_path=self.data_path, is_train=False, is_reshape=self.is_reshape,
                                      split_factor=self.split_factor,
                                      batch_size=self.batch_size)

        self.train_batches = self.train_dataset.num_batches
        self.val_batches = self.val_dataset.num_batches

        self.optimizer = keras.optimizers.Adam(learning_rate=1e-3)

        self.loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        self.loss_tracker = keras.metrics.Mean()
        self.val_loss_tracker = keras.metrics.Mean()

        self.train_acc = keras.metrics.SparseCategoricalAccuracy()
        self.val_acc = keras.metrics.SparseCategoricalAccuracy()

        self.acc_tracker = keras.metrics.Mean()
        self.val_acc_tracker = keras.metrics.Mean()

        # tensor board
        self.current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.train_log_dir = 'logs/' + self.current_time + '/train-' + self.model_type
        self.train_summary_writer = tf.summary.create_file_writer(self.train_log_dir)

        self.val_log_dir = 'logs/' + self.current_time + '/val-' + self.model_type
        self.val_summary_writer = tf.summary.create_file_writer(self.val_log_dir)

        # load pretrained weight:
        self.start_epoch = 0
        self.load_weights = False
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(self.start_epoch + 1, name="step"), optimizer=self.optimizer,
                                        net=self.model)
        self.manager = tf.train.CheckpointManager(self.ckpt, './tf_ckpts/' + self.model_type, max_to_keep=1)
        if self.load_weights:
            self.ckpt.restore(self.manager.latest_checkpoint)
            if self.manager.latest_checkpoint:
                print("Loading weights from {}".format(self.manager.latest_checkpoint))
            else:
                print("Loading weights failed. Initializing from scratch.")

    def get_model(self):
        if self.model_type == 'mlp':
            self.is_reshape = False
            inputs = keras.Input(shape=(784,), name="fashion_mlp")
            return mlp.create_model(inputs, name=self.model_type)
        elif self.model_type == 'mlp_2':
            self.is_reshape = False
            inputs = keras.Input(shape=(784,), name="fashion_mlp_2")
            return mlp.create_model_2(inputs, name=self.model_type)
        elif self.model_type == 'mlp_sm':
            self.is_reshape = False
            inputs = keras.Input(shape=(784,), name="fashion_mlp_sm")
            return mlp.create_model_sm(inputs, name=self.model_type)

        elif self.model_type == 'lenet' or self.model_type == 'lenet-64':
            self.is_reshape = True
            inputs = keras.Input(shape=(28, 28, 1), name="fashion_lenet")
            return lenet.create_model(inputs)
        elif self.model_type == 'lenet-dropout':
            self.is_reshape = True
            inputs = keras.Input(shape=(28, 28, 1), name="fashion_lenet_dropout")
            return lenet.create_model_do(inputs)
        elif self.model_type == 'lenet-bn-8' or self.model_type == 'lenet-bn-64':
            self.is_reshape = True
            inputs = keras.Input(shape=(28, 28, 1), name="fashion_bn")
            return lenet.create_model_bn(inputs)

        elif self.model_type == 'alexnet':
            self.is_reshape = True
            inputs = keras.Input(shape=(28, 28, 1), name="fashion-alexnet")
            return alexnet.create_model(inputs, name=self.model_type)
        elif self.model_type == 'alexnet_2':
            self.is_reshape = True
            inputs = keras.Input(shape=(28, 28, 1), name="fashion_alexnet_2")
            return alexnet.create_model_2(inputs, name=self.model_type)

        elif self.model_type == 'resnet':
            self.is_reshape = True
            inputs = keras.Input(shape=(28, 28, 1), name="fashion_restnet")
            return restnet.create_model(inputs, name=self.model_type)
        elif self.model_type == 'resnet_2':
            self.is_reshape = True
            inputs = keras.Input(shape=(28, 28, 1), name="fashion_restnet_2")
            return restnet.create_model_2(inputs, name=self.model_type)
        elif self.model_type == 'resnet_3':
            self.is_reshape = True
            inputs = keras.Input(shape=(28, 28, 1), name="fashion_restnet_3")
            return restnet.create_model_3(inputs, name=self.model_type)

        elif self.model_type == 'googlenet':
            self.is_reshape = True
            inputs = keras.Input(shape=(28, 28, 1), name="fashion_googlenet")
            return googlenet.create_model(inputs, name=self.model_type)
        elif self.model_type == 'googlenet_2':
            self.is_reshape = True
            inputs = keras.Input(shape=(28, 28, 1), name="fashion_googlenet_2")
            return googlenet.create_model_2(inputs, name=self.model_type)

    def train(self):
        keras.utils.plot_model(self.model, "{}.png".format(self.model_type), show_shapes=True)
        print('Train samples: {}'.format(self.train_dataset.num_samples))
        print('Val samples: {}'.format(self.val_dataset.num_samples))
        for epoch in range(self.start_epoch, self.epochs):
            print("\nStart training of epoch {}".format(epoch + 1))
            tb_i = tf.keras.utils.Progbar(self.train_batches, stateful_metrics=['loss', 'accuracy'])
            tb_i.update(0, values=0)
            for i, (x_batch_train, y_batch_train) in enumerate(self.train_dataset):
                with tf.GradientTape() as tape:
                    predicts = self.model(x_batch_train, training=True)
                    loss_value = self.loss_fn(y_batch_train, predicts)
                grads = tape.gradient(loss_value, self.model.trainable_weights)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

                self.train_acc.update_state(y_batch_train, predicts)

                self.loss_tracker.update_state(loss_value)
                self.acc_tracker.update_state(self.train_acc.result())
                values = [('loss', self.loss_tracker.result()), ('accuracy', self.acc_tracker.result())]
                tb_i.update(i + 1, values=values)
            with self.train_summary_writer.as_default():
                tf.summary.scalar('loss', self.loss_tracker.result(), step=epoch + 1)
                tf.summary.scalar('accuracy', self.acc_tracker.result(), step=epoch + 1)

            print("\nStart valid of epoch {}".format(epoch + 1))
            vb_i = tf.keras.utils.Progbar(self.val_batches, stateful_metrics=['val_loss', 'val_accuracy'])
            vb_i.update(0, values=0)
            for i, (x_batch_val, y_batch_val) in enumerate(self.val_dataset):
                val_predicts = self.model(x_batch_val, training=False)
                val_loss_value = self.loss_fn(y_batch_val, val_predicts)

                self.val_acc.update_state(y_batch_val, val_predicts)

                self.val_loss_tracker.update_state(val_loss_value)
                self.val_acc_tracker.update_state(self.val_acc.result())
                values = [('val_loss', self.val_loss_tracker.result()),
                          ('val_accuracy', self.val_acc_tracker.result())]
                vb_i.update(i + 1, values=values)
            with self.val_summary_writer.as_default():
                tf.summary.scalar('val_loss', self.val_loss_tracker.result(), step=epoch + 1)
                tf.summary.scalar('val_accuracy', self.val_acc_tracker.result(), step=epoch + 1)

            # save model
            save_path = self.manager.save()
            print('Saved at epoch {}'.format(epoch + 1))

            self.train_acc.reset_states()
            self.val_acc.reset_states()
            self.loss_tracker.reset_states()
            self.acc_tracker.reset_states()
            self.val_loss_tracker.reset_states()
            self.val_acc_tracker.reset_states()
