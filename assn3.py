# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Minimal script for reproducing the figures of the StyleGAN paper using pre-trained generators."""


import tensorflow as tf
import os
import pickle
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import config
import matplotlib.pyplot as plt

#----------------------------------------------------------------------------
# Helpers for loading and using pre-trained generators.

url       = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ' # karras2019stylegan-ffhq-1024x1024.pkl
# url_celebahq    = 'https://drive.google.com/uc?id=1MGqJl28pN4t7SAtSrPdSRJSQJqahkzUf' # karras2019stylegan-celebahq-1024x1024.pkl
# url_bedrooms    = 'https://drive.google.com/uc?id=1MOSKeGF0FJcivpBI7s63V9YHloUTORiF' # karras2019stylegan-bedrooms-256x256.pkl
# url_cars        = 'https://drive.google.com/uc?id=1MJ6iCfNtMIRicihwRorsM3b7mmtmK9c3' # karras2019stylegan-cars-512x384.pkl
# url_cats        = 'https://drive.google.com/uc?id=1MQywl0FNt6lHu8E_EUqnRbviagS7fbiJ' # karras2019stylegan-cats-256x256.pkl


synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True), minibatch_size=8)

_Gs_cache = dict()
# Load the StyleGAN generator network
def load_Gs(url):
    if url not in _Gs_cache:
        with dnnlib.util.open_url(url, cache_dir=config.cache_dir) as f:
            _G, _D, Gs = pickle.load(f)
        _Gs_cache[url] = Gs
    return _Gs_cache[url]

tflib.init_tf()
os.makedirs(config.result_dir, exist_ok=True)

#----------------------------------------------------------------------------
### Part - 1

Gs = load_Gs(url)

# Print the architecture of the generator network
Gs.print_layers()

# Choose a layer of interest to visualize and print its weights
layer_idx = 0
layer = Gs.layers[layer_idx]
print(layer)
weights_0 = layer.get_weight()
print(weights_0)

#----------------------------------------------------------------------------
### Part - 2

# Choose a layer to perform style mixing at
mixing_layer_idx = 4

# Generate two random latent vectors
latent_size = Gs.input_shape[1]
latent_vector_1 = np.random.randn(1, latent_size)
latent_vector_2 = np.random.randn(1, latent_size)

# Perform style mixing between the two latent vectors at the chosen layer
w_1 = Gs.components.mapping(tf.constant(latent_vector_1), None)
w_2 = Gs.components.mapping(tf.constant(latent_vector_2), None)
w_mix = tf.concat([w_1[:, :mixing_layer_idx], w_2[:, mixing_layer_idx:]], axis=1)

# Generate an image from the mixed latent vector
img = Gs.components.synthesis(w_mix)
img = (img + 1) * 0.5  # Scale the pixel values to the range [0, 1]

# Display the resulting image
plt.imshow(img[0])
plt.axis("off")
plt.show()


#----------------------------------------------------------------------------
### Part - 3
# Choose another layer of interest to visualize and print its weights
layer_idx_2 = 3
layer = Gs.layers[layer_idx_2]
print(layer)
weights_3 = layer.get_weight()
print(weights_3)

# The mean absolute difference between the two sets of weights
mean_abs_diff = tf.reduce_mean(tf.abs(weights_0 - weights_3))
print("Mean absolute difference: %.6f" % mean_abs_diff.numpy())

#----------------------------------------------------------------------------
### Part - 4
####### Mentioned in Report

#----------------------------------------------------------------------------
### Part - 5
####### Mentioned in Report


#----------------------------------------------------------------------------
### Part - 6

def draw_style_mixing_figure(png, Gs, w, h, src_seeds, dst_seeds, style_ranges):
    print(png)
    src_latents = np.stack(np.random.RandomState(seed).randn(Gs.input_shape[1]) for seed in src_seeds)
    dst_latents = np.stack(np.random.RandomState(seed).randn(Gs.input_shape[1]) for seed in dst_seeds)
    src_dlatents = Gs.components.mapping.run(src_latents, None) # [seed, layer, component]
    dst_dlatents = Gs.components.mapping.run(dst_latents, None) # [seed, layer, component]
    src_images = Gs.components.synthesis.run(src_dlatents, randomize_noise=False, **synthesis_kwargs)
    dst_images = Gs.components.synthesis.run(dst_dlatents, randomize_noise=False, **synthesis_kwargs)

    canvas = PIL.Image.new('RGB', (w * (len(src_seeds) + 1), h * (len(dst_seeds) + 1)), 'white')
    for col, src_image in enumerate(list(src_images)):
        canvas.paste(PIL.Image.fromarray(src_image, 'RGB'), ((col + 1) * w, 0))
    for row, dst_image in enumerate(list(dst_images)):
        canvas.paste(PIL.Image.fromarray(dst_image, 'RGB'), (0, (row + 1) * h))
        row_dlatents = np.stack([dst_dlatents[row]] * len(src_seeds))
        row_dlatents[:, style_ranges[row]] = src_dlatents[:, style_ranges[row]]
        row_images = Gs.components.synthesis.run(row_dlatents, randomize_noise=False, **synthesis_kwargs)
        for col, image in enumerate(list(row_images)):
            canvas.paste(PIL.Image.fromarray(image, 'RGB'), ((col + 1) * w, (row + 1) * h))
    canvas.save(png)

draw_style_mixing_figure(os.path.join(config.result_dir, 'figure03-style-mixing.png'), load_Gs(url), w=1024, h=1024, src_seeds=[639,701,687,615,2268], dst_seeds=[888,829,1898,1733,1614,845], style_ranges=[range(0,4)]*3+[range(4,8)]*2+[range(8,18)])


