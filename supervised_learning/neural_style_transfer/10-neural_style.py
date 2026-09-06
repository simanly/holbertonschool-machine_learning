#!/usr/bin/env python3
"""Neural Style Transfer Module with Variational Cost"""

import numpy as np
import tensorflow as tf


class NST:
    """Class NST that performs Neural Style Transfer"""

    style_layers = [
        'block1_conv1',
        'block2_conv1',
        'block3_conv1',
        'block4_conv1',
        'block5_conv1'
    ]
    content_layer = 'block4_conv2'

    def __init__(self, style_image, content_image,
                 alpha=1e4, beta=1, var=10):
        """Initializes the NST class instance"""
        if (not isinstance(style_image, np.ndarray)
                or style_image.ndim != 3 or style_image.shape[2] != 3):
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)"
            )

        if (not isinstance(content_image, np.ndarray)
                or content_image.ndim != 3 or content_image.shape[2] != 3):
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)"
            )

        if not isinstance(alpha, (int, float)) or isinstance(alpha, bool) \
                or alpha < 0:
            raise TypeError("alpha must be a non-negative number")

        if not isinstance(beta, (int, float)) or isinstance(beta, bool) \
                or beta < 0:
            raise TypeError("beta must be a non-negative number")

        if not isinstance(var, (int, float)) or isinstance(var, bool) \
                or var < 0:
            raise TypeError("var must be a non-negative number")

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.var = var
        self.load_model()
        self.generate_features()

    @staticmethod
    def scale_image(image, max_dim=512):
        """Rescales an image so that its maximum dimension is max_dim"""
        if (not isinstance(image, np.ndarray)
                or image.ndim != 3 or image.shape[2] != 3):
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)"
            )

        if type(max_dim) is not int or max_dim <= 0:
            raise TypeError("max_dim must be a positive integer")

        h, w, _ = image.shape
        if h > w:
            h_new = max_dim
            w_new = int((w * max_dim) / h)
        else:
            w_new = max_dim
            h_new = int((h * max_dim) / w)

        image_scaled = image / 255.0
        image_resized = tf.image.resize(
            image_scaled,
            size=[h_new, w_new],
            method='bicubic'
        )
        image_resized = tf.clip_by_value(image_resized, 0.0, 1.0)

        return tf.expand_dims(image_resized, axis=0)

    def load_model(self):
        """Loads VGG19 model and sets up the outputs for style and content layers"""
        # 1. Загружаем VGG19 без верхних слоев
        vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        
        # Замораживаем веса базовой модели
        vgg.trainable = False

        # 2. Список всех целевых слоев
        style_nodes = self.style_layers
        content_node = [self.content_layer]
        target_layers = style_nodes + content_node

        # 3. Реконструируем граф с заменой MaxPooling -> AveragePooling
        x = vgg.input
        outputs = []

        # Проходим по всем слоям, кроме InputLayer (vgg.layers[0])
        for layer in vgg.layers[1:]:
            if isinstance(layer, tf.keras.layers.MaxPooling2D):
                # Заменяем MaxPooling на AveragePooling с теми же параметрами
                x = tf.keras.layers.AveragePooling2D(
                    pool_size=layer.pool_size,
                    strides=layer.strides,
                    padding=layer.padding,
                    name=layer.name
                )(x)
            else:
                layer.trainable = False
                x = layer(x)

            # Собираем выходы нужных слоев
            if layer.name in target_layers:
                outputs.append(x)

        # 4. Создаем итоговую модель
        # Входы и выходы должны принадлежать ОДНОМУ и тому же графу (x)
        self.model = tf.keras.models.Model(inputs=vgg.input, outputs=outputs)

    @staticmethod
    def gram_matrix(input_tensor):
        """Calculates the Gram matrix of a tensor"""
        if not isinstance(input_tensor, (tf.Tensor, tf.Variable)) or \
                input_tensor.ndim != 4:
            raise TypeError("input_tensor must be a tensor of rank 4")

        channels = int(input_tensor.shape[-1])
        a = tf.reshape(input_tensor, [-1, channels])
        n = tf.cast(tf.shape(a)[0], tf.float32)
        gram = tf.matmul(a, a, transpose_a=True)

        return gram / n

    def generate_features(self):
        """Extracts style and content features from reference images"""
        style_preprocessed = tf.keras.applications.vgg19.preprocess_input(
            self.style_image * 255.0
        )
        content_preprocessed = tf.keras.applications.vgg19.preprocess_input(
            self.content_image * 255.0
        )

        style_outputs = self.model(style_preprocessed)[:-1]
        content_outputs = self.model(content_preprocessed)[-1]

        self.gram_style_features = [
            self.gram_matrix(style_layer) for style_layer in style_outputs
        ]
        self.content_feature = content_outputs

    def layer_style_cost(self, style_output, gram_style):
        """Calculates the style cost for a single layer"""
        if not isinstance(style_output, (tf.Tensor, tf.Variable)) or \
                style_output.ndim != 4:
            raise TypeError("style_output must be a tensor of rank 4")

        c = style_output.shape[-1]
        if not isinstance(gram_style, (tf.Tensor, tf.Variable)) or \
                gram_style.shape != (c, c):
            raise TypeError(
                f"gram_style must be a tensor of shape ({c}, {c})"
            )

        gram_style_output = self.gram_matrix(style_output)
        return tf.reduce_mean(tf.square(gram_style_output - gram_style))

    def style_cost(self, style_outputs):
        """Calculates the style cost for all layers"""
        if type(style_outputs) is not list or \
                len(style_outputs) != len(self.style_layers):
            raise TypeError(
                f"style_outputs must be a list with a length of "
                f"{len(self.style_layers)}"
            )

        weight_per_layer = 1.0 / float(len(self.style_layers))
        J_style = 0.0

        for style_output, gram_style in zip(
                style_outputs, self.gram_style_features):
            J_style += weight_per_layer * self.layer_style_cost(
                style_output, gram_style
            )

        return J_style

    def content_cost(self, content_output):
        """Calculates the content cost"""
        if not isinstance(content_output, (tf.Tensor, tf.Variable)) or \
                content_output.shape != self.content_feature.shape:
            raise TypeError(
                "content_output must be a tensor of shape "
                f"{self.content_feature.shape}"
            )

        return tf.reduce_mean(
            tf.square(content_output - self.content_feature)
        ) / 2.0

    @staticmethod
    def variational_cost(image):
        """Calculates the variational cost for the generated image"""
        if not isinstance(image, (tf.Tensor, tf.Variable)) or \
                len(image.shape) not in (3, 4):
            raise TypeError("image must be a tensor of rank 3 or 4")

        return tf.image.total_variation(image)

    def total_cost(self, generated_image):
        """Calculates the total cost for the generated image"""
        if not isinstance(generated_image, (tf.Tensor, tf.Variable)) or \
                generated_image.shape != self.content_image.shape:
            raise TypeError(
                "generated_image must be a tensor of shape "
                f"{self.content_image.shape}"
            )

        gen_preprocessed = tf.keras.applications.vgg19.preprocess_input(
            generated_image * 255.0
        )
        outputs = self.model(gen_preprocessed)

        style_outputs = outputs[:-1]
        content_output = outputs[-1]

        J_content = self.content_cost(content_output)
        J_style = self.style_cost(style_outputs)
        J_var = self.variational_cost(generated_image)

        J_total = (self.alpha * J_content) + (self.beta * J_style) + \
                  (self.var * J_var)

        return J_total, J_content, J_style, J_var

    def compute_grads(self, generated_image):
        """Calculates gradients for the generated image"""
        if not isinstance(generated_image, (tf.Tensor, tf.Variable)) or \
                generated_image.shape != self.content_image.shape:
            raise TypeError(
                "generated_image must be a tensor of shape "
                f"{self.content_image.shape}"
            )

        with tf.GradientTape() as tape:
            J_total, J_content, J_style, J_var = self.total_cost(
                generated_image
            )

        grads = tape.gradient(J_total, generated_image)
        return grads, J_total, J_content, J_style, J_var

    def generate_image(self, iterations=1000, step=None, lr=0.01,
                       beta1=0.9, beta2=0.99):
        """Generates the neural style transferred image"""
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be positive")

        if step is not None:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step <= 0 or step >= iterations:
                raise ValueError(
                    "step must be positive and less than iterations"
                )

        if not isinstance(lr, (int, float)) or isinstance(lr, bool):
            raise TypeError("lr must be a number")
        if lr <= 0:
            raise ValueError("lr must be positive")

        if type(beta1) is not float:
            raise TypeError("beta1 must be a float")
        if not (0.0 <= beta1 <= 1.0):
            raise ValueError("beta1 must be in the range [0, 1]")

        if type(beta2) is not float:
            raise TypeError("beta2 must be a float")
        if not (0.0 <= beta2 <= 1.0):
            raise ValueError("beta2 must be in the range [0, 1]")

        generated_image = tf.Variable(self.content_image)
        optimizer = tf.optimizers.Adam(
            learning_rate=lr, beta_1=beta1, beta_2=beta2
        )

        best_cost = float('inf')
        best_image = None

        for i in range(iterations + 1):
            grads, J_total, J_content, J_style, J_var = self.compute_grads(
                generated_image
            )

            if J_total < best_cost:
                best_cost = J_total
                best_image = generated_image[0]

            if step is not None and (i % step == 0 or i == iterations):
                print(
                    f"Cost at iteration {i}: {J_total}, "
                    f"content {J_content}, style {J_style}, var {J_var}"
                )

            if i < iterations:
                optimizer.apply_gradients([(grads, generated_image)])
                generated_image.assign(
                    tf.clip_by_value(generated_image, 0.0, 1.0)
                )

        return best_image, best_cost
