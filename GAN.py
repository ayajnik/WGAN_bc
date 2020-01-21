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
        self.logits = Dense(units=1, activation="sigmoid", kernel_initializer=init)

    def call(self, inputs):
        x = self.input_layer(inputs)
        x = self.hid_1(x)
        return self.logits(x)


@tf.function
def train_step(real_data, gen, disc, noise_dim, generator_optimizer, discriminator_optimizer):

    batch_size = real_data.shape[0]

    noise = tf.random.normal([batch_size, noise_dim])
    
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        fake_data = gen(noise, training=True)

        real_output = disc(real_data, training=True)
        fake_output = disc(fake_data, training=True)

        gen_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
        disc_loss = cross_entropy(tf.ones_like(real_output), real_output) + cross_entropy(tf.zeros_like(fake_output), fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, gen.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, disc.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, gen.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, disc.trainable_variables))

    return gen_loss, disc_loss





