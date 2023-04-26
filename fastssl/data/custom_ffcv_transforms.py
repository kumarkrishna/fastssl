import numpy as np
from numpy.random import rand
from typing import Callable, Optional, Tuple
from dataclasses import replace
from ffcv.pipeline.allocation_query import AllocationQuery
from ffcv.pipeline.operation import Operation
from ffcv.pipeline.state import State
from ffcv.pipeline.compiler import Compiler
import numbers
import numba as nb
from scipy.signal import convolve2d
import random


class ColorJitter(Operation):
    """Add ColorJitter with probability jitter_prob.
    Operates on raw arrays (not tensors). Note that the values should be between 0 and 255
    Parameters
    ----------
    jitter_prob : float, The probability with which to apply ColorJitter.
    brightness (float or tuple of float (min, max)): How much to jitter brightness.
        brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
        or the given [min, max]. Should be non negative numbers.
    contrast (float or tuple of float (min, max)): How much to jitter contrast.
        contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
        or the given [min, max]. Should be non negative numbers.
    saturation (float or tuple of float (min, max)): How much to jitter saturation.
        saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
        or the given [min, max]. Should be non negative numbers.
    hue (float or tuple of float (min, max)): How much to jitter hue.
        hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
        Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """

    def __init__(self, jitter_prob, brightness=0, contrast=0, saturation=0, hue=0):
        super().__init__()
        self.brightness = self._check_input(brightness, "brightness")
        self.contrast = self._check_input(contrast, "contrast")
        self.saturation = self._check_input(saturation, "saturation")
        self.hue = self._check_input(
            hue, "hue", center=0, bound=(-0.5, 0.5), clip_first_on_zero=False
        )
        self.jitter_prob = jitter_prob
        assert self.jitter_prob >= 0 and self.jitter_prob <= 1

    def _check_input(
        self, value, name, center=1, bound=(0, float("inf")), clip_first_on_zero=True
    ):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError(
                    f"If {name} is a single number, it must be non negative."
                )
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError(f"{name} values should be between {bound}")
        else:
            raise TypeError(
                f"{name} should be a single number or a list/tuple with length 2."
            )

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    def generate_code(self) -> Callable:
        my_range = Compiler.get_iterator()
        jitter_prob = self.jitter_prob
        apply_brightness = self.brightness is not None
        if apply_brightness:
            brightness_min, brightness_max = self.brightness
        apply_contrast = self.contrast is not None
        if apply_contrast:
            contrast_min, contrast_max = self.contrast
        apply_saturation = self.saturation is not None
        if apply_saturation:
            saturation_min, saturation_max = self.saturation
        apply_hue = self.hue is not None
        if apply_hue:
            hue_min, hue_max = self.hue

        def color_jitter(images, dst):
            should_jitter = rand(images.shape[0]) < jitter_prob
            for i in my_range(images.shape[0]):
                if should_jitter[i]:
                    img = images[i]
                    # Brightness
                    if apply_brightness:
                        ratio_brightness = np.random.uniform(
                            brightness_min, brightness_max
                        )
                        img = (
                            ratio_brightness * img + (1.0 - ratio_brightness) * img * 0
                        )
                        img = np.clip(img, 0, 255)

                    # Contrast
                    if apply_contrast:
                        ratio_contrast = np.random.uniform(contrast_min, contrast_max)
                        gray = (
                            0.2989 * img[:, :, 0:1]
                            + 0.5870 * img[:, :, 1:2]
                            + 0.1140 * img[:, :, 2:3]
                        )
                        img = (
                            ratio_contrast * img + (1.0 - ratio_contrast) * gray.mean()
                        )
                        img = np.clip(img, 0, 255)

                    # Saturation
                    if apply_saturation:
                        ratio_saturation = np.random.uniform(
                            saturation_min, saturation_max
                        )
                        dst[i] = (
                            0.2989 * img[:, :, 0:1]
                            + 0.5870 * img[:, :, 1:2]
                            + 0.1140 * img[:, :, 2:3]
                        )
                        img = ratio_saturation * img + (1.0 - ratio_saturation) * dst[i]
                        img = np.clip(img, 0, 255)

                    # Hue
                    if apply_hue:
                        img = img / 255.0
                        hue_factor = np.random.uniform(hue_min, hue_max)
                        hue_factor_radians = hue_factor * 2.0 * np.pi
                        cosA = np.cos(hue_factor_radians)
                        sinA = np.sin(hue_factor_radians)
                        hue_rotation_matrix = [
                            [
                                cosA + (1.0 - cosA) / 3.0,
                                1.0 / 3.0 * (1.0 - cosA) - np.sqrt(1.0 / 3.0) * sinA,
                                1.0 / 3.0 * (1.0 - cosA) + np.sqrt(1.0 / 3.0) * sinA,
                            ],
                            [
                                1.0 / 3.0 * (1.0 - cosA) + np.sqrt(1.0 / 3.0) * sinA,
                                cosA + 1.0 / 3.0 * (1.0 - cosA),
                                1.0 / 3.0 * (1.0 - cosA) - np.sqrt(1.0 / 3.0) * sinA,
                            ],
                            [
                                1.0 / 3.0 * (1.0 - cosA) - np.sqrt(1.0 / 3.0) * sinA,
                                1.0 / 3.0 * (1.0 - cosA) + np.sqrt(1.0 / 3.0) * sinA,
                                cosA + 1.0 / 3.0 * (1.0 - cosA),
                            ],
                        ]
                        hue_rotation_matrix = np.array(
                            hue_rotation_matrix, dtype=img.dtype
                        )
                        for row in nb.prange(img.shape[0]):
                            for col in nb.prange(img.shape[1]):
                                r, g, b = img[row, col, :]
                                img[row, col, 0] = (
                                    r * hue_rotation_matrix[0, 0]
                                    + g * hue_rotation_matrix[0, 1]
                                    + b * hue_rotation_matrix[0, 2]
                                )
                                img[row, col, 1] = (
                                    r * hue_rotation_matrix[1, 0]
                                    + g * hue_rotation_matrix[1, 1]
                                    + b * hue_rotation_matrix[1, 2]
                                )
                                img[row, col, 2] = (
                                    r * hue_rotation_matrix[2, 0]
                                    + g * hue_rotation_matrix[2, 1]
                                    + b * hue_rotation_matrix[2, 2]
                                )
                        img = np.asarray(np.clip(img * 255.0, 0, 255), dtype=np.uint8)
                    dst[i] = img
                else:
                    dst[i] = images[i]
            return dst

        color_jitter.is_parallel = True
        return color_jitter

    def declare_state_and_memory(
        self, previous_state: State
    ) -> Tuple[State, Optional[AllocationQuery]]:
        return (
            replace(previous_state, jit_mode=True),
            AllocationQuery(shape=previous_state.shape, dtype=previous_state.dtype),
        )


class RandomGrayscale(Operation):
    """Randomly convert image to grayscale with a probability of p (default 0.1).
    Operates on raw arrays (not tensors).

    Args:
        p (float): probability that image should be converted to grayscale.

    """

    def __init__(self, p=0.1):
        super().__init__()
        self.p = p
        assert (
            self.p >= 0 and self.p <= 1
        ), "RandomGrayscale p should be between 0 and 1, currently set to {}".format(
            self.p
        )

    def generate_code(self) -> Callable:
        my_range = Compiler.get_iterator()
        grayscale_prob = self.p

        def random_grayscale(images, dst):
            should_grayscale = rand(images.shape[0]) < grayscale_prob
            for i in my_range(images.shape[0]):
                if should_grayscale[i]:
                    img = np.asarray(images[i], dtype=np.float32)
                    dst[i] = (
                        0.2989 * img[:, :, 0:1]
                        + 0.5870 * img[:, :, 1:2]
                        + 0.1140 * img[:, :, 2:3]
                    )
                    # print(img.dtype,dst[i].dtype,dst[i].shape,dst[i].mean())
                else:
                    dst[i] = images[i]
            return dst

        random_grayscale.is_parallel = True
        return random_grayscale

    def declare_state_and_memory(
        self, previous_state: State
    ) -> Tuple[State, Optional[AllocationQuery]]:
        return (
            replace(previous_state, jit_mode=True),
            AllocationQuery(shape=previous_state.shape, dtype=previous_state.dtype),
        )


def apply_blur(img, kernel_size, w):
    pad = (kernel_size - 1) // 2
    H, W, _ = img.shape
    tmp = np.zeros(img.shape, dtype=np.float32)
    for k in range(kernel_size):
        start = max(0, pad - k)
        stop = min(W, pad - k + W)
        window = (img[:, start:stop] / 255) * w[k]
        tmp[:, np.abs(stop - W) : W - start] += window
    tmp2 = tmp + 0.0
    for k in range(kernel_size):
        start = max(0, pad - k)
        stop = min(H, pad - k + H)
        window = (tmp[start:stop] * w[k]).astype(np.uint8)
        tmp2[np.abs(stop - H) : H - start] += window
    return np.clip(tmp2 * 255.0, 0, 255).astype(np.uint8)


class GaussianBlur(Operation):
    # credits: https://github.com/facebookresearch/FFCV-SSL/tree/main/ffcv/transforms
    """Blurs image with randomly chosen Gaussian blur.
    If the image is torch Tensor, it is expected
    to have [..., C, H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        blur_prob (float): probability to apply blurring to each input
        kernel_size (int or sequence): Size of the Gaussian kernel.
        sigma (float or tuple of float (min, max)): Standard deviation to be used for
            creating kernel to perform blurring. If float, sigma is fixed. If it is tuple
            of float (min, max), sigma is chosen uniformly at random to lie in the
            given range.
    """

    def __init__(self, blur_prob, kernel_size=5, sigma=(0.1, 2.0), seed=None):
        super().__init__()
        self.blur_prob = blur_prob
        self.kernel_size = kernel_size
        assert sigma[1] > sigma[0]
        self.sigmas = np.linspace(sigma[0], sigma[1], 10)
        from scipy import signal

        self.weights = np.stack(
            [
                signal.gaussian(kernel_size, s)
                for s in np.linspace(sigma[0], sigma[1], 10)
            ]
        )
        self.weights /= self.weights.sum(1, keepdims=True)
        self.seed = seed

    def generate_code(self) -> Callable:
        my_range = Compiler.get_iterator()
        blur_prob = self.blur_prob
        kernel_size = self.kernel_size
        weights = self.weights
        seed = self.seed
        apply_blur_c = Compiler.compile(apply_blur)

        def blur(images, _, indices):
            for i in my_range(images.shape[0]):
                if np.random.rand() < blur_prob:
                    k = np.random.randint(low=0, high=10)
                    for ch in range(images.shape[-1]):
                        images[i, ..., ch] = convolve2d(
                            images[i, ..., ch],
                            np.outer(weights[k], weights[k]),
                            mode="same",
                        )
                    # images[i] = apply_blur_c(images[i], kernel_size, weights[k])
            return images

        blur.is_parallel = True
        blur.with_indices = True
        return blur

    def declare_state_and_memory(
        self, previous_state: State
    ) -> Tuple[State, Optional[AllocationQuery]]:
        return (
            replace(previous_state, jit_mode=False),
            None,
        )


class RandomSolarization(Operation):
    # credits: https://github.com/facebookresearch/FFCV-SSL/tree/main/ffcv/transforms
    """Solarize the image randomly with a given probability by inverting all pixel
    values above a threshold. If img is a Tensor, it is expected to be in [..., 1 or 3, H, W] format,
    where ... means it can have an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".
    Parameters
    ----------
        solarization_prob (float): probability of the image being solarized. Default value is 0.5
        threshold (float): all pixels equal or above this value are inverted.
    """

    def __init__(
        self, solarization_prob: float = 0.5, threshold: float = 128, seed: int = None
    ):
        super().__init__()
        self.sol_prob = solarization_prob
        self.threshold = threshold
        self.seed = seed

    def generate_code(self) -> Callable:
        my_range = Compiler.get_iterator()
        sol_prob = self.sol_prob
        threshold = self.threshold
        seed = self.seed

        if seed is None:

            def solarize(images, _):
                for i in my_range(images.shape[0]):
                    if np.random.rand() < sol_prob:
                        mask = images[i] >= threshold
                        images[i] = np.where(mask, 255 - images[i], images[i])
                return images

            solarize.is_parallel = True
            return solarize

        def solarize(images, _, counter):
            random.seed(seed + counter)
            values = np.zeros(len(images))
            for i in range(len(images)):
                values[i] = random.uniform(0, 1)
            for i in my_range(images.shape[0]):
                if values[i] < sol_prob:
                    mask = images[i] >= threshold
                    images[i] = np.where(mask, 255 - images[i], images[i])
            return images

        solarize.with_counter = True
        solarize.is_parallel = True
        return solarize

    def declare_state_and_memory(
        self, previous_state: State
    ) -> Tuple[State, Optional[AllocationQuery]]:
        return (replace(previous_state, jit_mode=True), None)
