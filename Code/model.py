import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

#Configure VAE

#Set up parameters

#learning_rate = 0.01
learning_rate = 0.1
num_steps = 100

display_step = 20
examples_to_show =5

# Network Parameters
# num_hidden_1 = 256 # 1st layer num features
# num_hidden_2 = 128 # 2nd layer num features (the latent dim)
num_input = XA.shape[1] 

# Network Parameters
hidden_dim = XA.shape[1]
latent_dim = XA.shape[1]

def glorot_init(shape):
    return tf.random_normal(shape=shape, stddev=1. / tf.sqrt(shape[0] / 2.))

#Configure layers
# Variables
input_data = tf.placeholder("float", [None, num_input])

weights = {
    'encoder_h1': tf.Variable(glorot_init([num_input, hidden_dim])),
    'encoder_h2': tf.Variable(glorot_init([num_input, hidden_dim])),
    'z_mean': tf.Variable(glorot_init([hidden_dim, latent_dim])),    
    'z_std': tf.Variable(glorot_init([hidden_dim, latent_dim])),
    'decoder_h1': tf.Variable(glorot_init([latent_dim, hidden_dim])),
    'decoder_h2': tf.Variable(glorot_init([latent_dim, hidden_dim])),
    'decoder_out': tf.Variable(glorot_init([hidden_dim, num_input]))
}
biases = {
    'encoder_b1': tf.Variable(glorot_init([hidden_dim])),
    'encoder_b2': tf.Variable(glorot_init([hidden_dim])),
    'z_mean': tf.Variable(glorot_init([latent_dim])),
    'z_std': tf.Variable(glorot_init([latent_dim])),
    'decoder_b1': tf.Variable(glorot_init([hidden_dim])),
    'decoder_b2': tf.Variable(glorot_init([hidden_dim])),
    'decoder_out': tf.Variable(glorot_init([num_input]))
}


# Building the encoder
def encoder(x):   
    encoder = tf.matmul(x, weights['encoder_h1']) + biases['encoder_b1']
    encoder = tf.nn.tanh(encoder)
    encoder = tf.matmul(encoder, weights['encoder_h2']) + biases['encoder_b2']
    encoder = tf.nn.tanh(encoder)
    z_mean = tf.matmul(encoder, weights['z_mean']) + biases['z_mean']
    z_std = tf.matmul(encoder, weights['z_std']) + biases['z_std']

    # Sampler: Normal (gaussian) random distribution
    eps = tf.random_normal(tf.shape(z_std), dtype=tf.float32, mean=0., stddev=1.0,
                               name='epsilon')
    z = z_mean + tf.exp(z_std / 2) * eps

    return z_mean, z_std, z, encoder

# Building the decoder (with scope to re-use these layers later)
def decoder(z):
    decoder = tf.matmul(z, weights['decoder_h1']) + biases['decoder_b1']
    decoder = tf.nn.tanh(decoder)
    decoder = tf.matmul(z, weights['decoder_h2']) + biases['decoder_b2']
    decoder = tf.nn.tanh(decoder)
    decoder = tf.matmul(decoder, weights['decoder_out']) + biases['decoder_out']
    decoder = tf.nn.sigmoid(decoder)

    return decoder

