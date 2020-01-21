import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Dense

class Generator(tf.keras.Model):

    def __init__(self, n_inp, n_noise, n_hid=128):
        super().__init__()
        init = tf.keras.initializers.GlorotUniform
        self.input_layer = Dense(units=n_noise, kernel_initializer=init)
        self.hid_1 = Dense(units=n_hid, activation="tanh", kernel_initializer=init)
        self.output_layer = Dense(units=n_inp, activation="tanh", kernel_initializer=init)

    def call(self, inputs):
        x = self.input_layer(inputs)
        x = self.hid_1(x)
        return self.output_layer(x)


class Discriminator(tf.keras.Model):

    def __init__(self, n_inp, n_hid=128):
        super().__init__()
        init = tf.keras.initializers.GlorotUniform
        self.input_layer = Dense(units=n_inp, kernel_initializer=init)
        self.hid_1 = Dense(units=n_hid, activation="tanh", kernel_initializer=init)
        self.logits = Dense(units=1, activation=None, kernel_initializer=init)

    def call(self, inputs):
        x = self.input_layer(inputs)
        x = self.hid_1(x)
        return self.logits(x)


@tf.function
def train_step(real_data, gen, disc, noise_dim, generator_optimizer, discriminator_optimizer):
    batch_size = real_data.shape[0]

    noise = tf.random.normal([batch_size, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        fake_data = gen(noise, training=True)

        real_output = disc(real_data, training=True)
        fake_output = disc(fake_data, training=True)

        disc_loss =   tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)
        gen_loss  = - tf.reduce_mean(fake_output)

    wasserstein = tf.reduce_mean(real_output) - tf.reduce_mean(fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, gen.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, disc.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, gen.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, disc.trainable_variables))
        
    tf.group(*(var.assign(tf.clip_by_value(var, -0.01, 0.01)) for var in disc.trainable_variables))

    return wasserstein





