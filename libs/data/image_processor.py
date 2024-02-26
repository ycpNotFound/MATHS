# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Image processor class for Donut."""

from typing import Dict, List, Optional, Union

import numpy as np
import PIL
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from transformers.image_transforms import (
    get_resize_output_image_size,
    normalize,
    pad,
    rescale,
    resize,
    to_channel_dimension_format,
)
from transformers.image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    get_image_size,
    # make_list_of_images,
    to_numpy_array,
    valid_images,
)
from transformers.utils import (
    TensorType, logging,
    ExplicitEnum,
    is_jax_tensor,
    is_tf_tensor,
    is_torch_available,
    is_torch_tensor,
    is_vision_available,
    requires_backends,
    to_numpy,
)
from transformers.utils.import_utils import is_vision_available


logger = logging.get_logger(__name__)

def is_valid_image(img):
    return (
        (is_vision_available() and isinstance(img, PIL.Image.Image))
        or isinstance(img, np.ndarray)
        or is_torch_tensor(img)
        or is_tf_tensor(img)
        or is_jax_tensor(img)
    )

if is_vision_available():
    import PIL

def is_batched(img):
    if isinstance(img, (list, tuple)):
        return is_valid_image(img[0])
    return False

def make_list_of_images(images, expected_ndims: int = 3) -> List[ImageInput]:
    """
    Ensure that the input is a list of images. If the input is a single image, it is converted to a list of length 1.
    If the input is a batch of images, it is converted to a list of images.

    Args:
        images (`ImageInput`):
            Image of images to turn into a list of images.
        expected_ndims (`int`, *optional*, defaults to 3):
            Expected number of dimensions for a single input image. If the input image has a different number of
            dimensions, an error is raised.
    """
    if is_batched(images):
        return images

    # Either the input is a single image, in which case we create a list of length 1
    if isinstance(images, PIL.Image.Image):
        # PIL images are never batched
        return [images]

    if is_valid_image(images):
        if images.ndim == expected_ndims + 1:
            # Batch of images
            images = [image for image in images]
        elif images.ndim == expected_ndims:
            # Single image
            images = [images]
        else:
            raise ValueError(
                f"Invalid image shape. Expected either {expected_ndims + 1} or {expected_ndims} dimensions, but got"
                f" {images.ndim} dimensions."
            )
        return images
    raise ValueError(
        "Invalid image type. Expected either PIL.Image.Image, numpy.ndarray, torch.Tensor, tf.Tensor or "
        f"jax.ndarray, but got {type(images)}."
    )

class DonutImageProcessor(BaseImageProcessor):
    r"""
    Constructs a Donut image processor.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions to the specified `size`. Can be overridden by
            `do_resize` in the `preprocess` method.
        size (`Dict[str, int]` *optional*, defaults to `{"shortest_edge": 224}`):
            Size of the image after resizing. The shortest edge of the image is resized to size["shortest_edge"], with
            the longest edge resized to keep the input aspect ratio. Can be overridden by `size` in the `preprocess`
            method.
        resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BILINEAR`):
            Resampling filter to use if resizing the image. Can be overridden by `resample` in the `preprocess` method.
        do_center_crop (`bool`, *optional*, defaults to `True`):
            Whether to center crop the image to the specified `crop_size`. Can be overridden by `do_center_crop` in the
            `preprocess` method.
        crop_size (`Dict[str, int]` *optional*, defaults to 224):
            Size of the output image after applying `center_crop`. Can be overridden by `crop_size` in the `preprocess`
            method.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by `do_rescale` in
            the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image. Can be overridden by `rescale_factor` in the `preprocess`
            method.
        do_normalize:
            Whether to normalize the image. Can be overridden by `do_normalize` in the `preprocess` method.
        image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_STD`):
            Image standard deviation.
        do_convert_rgb (`bool`, *optional*, defaults to `True`):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize: bool = True,
        size: Dict[str, int] = None,
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        do_thumbnail: bool = True,
        do_align_long_axis: bool = False,
        do_pad: bool = True,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        size = size if size is not None else {"height": 2560, "width": 1920}
        if isinstance(size, (tuple, list)):
            # The previous feature extractor size parameter was in (width, height) format
            size = size[::-1]
        size = get_size_dict(size)

        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.do_thumbnail = do_thumbnail
        self.do_align_long_axis = do_align_long_axis
        self.do_pad = do_pad
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else IMAGENET_STANDARD_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_STANDARD_STD

    def align_long_axis(
        self, image: np.ndarray, size: Dict[str, int], data_format: Optional[Union[str, ChannelDimension]] = None
    ) -> np.ndarray:
        """
        Align the long axis of the image to the longest axis of the specified size.

        Args:
            image (`np.ndarray`):
                The image to be aligned.
            size (`Dict[str, int]`):
                The size `{"height": h, "width": w}` to align the long axis to.

        Returns:
            `np.ndarray`: The aligned image.
        """
        input_height, input_width = get_image_size(image)
        output_height, output_width = size["height"], size["width"]

        if (output_width < output_height and input_width > input_height) or (
            output_width > output_height and input_width < input_height
        ):
            image = np.rot90(image, 3)

        if data_format is not None:
            image = to_channel_dimension_format(image, data_format)

        return image

    def rotate_image(self, *args, **kwargs):
        logger.info(
            "rotate_image is deprecated and will be removed in version 4.27. Please use align_long_axis instead."
        )
        return self.align_long_axis(*args, **kwargs)

    def pad_image(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        random_padding: bool = False,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        bboxes = None,
    ) -> np.ndarray:
        """
        Pad the image to the specified size.

        Args:
            image (`np.ndarray`):
                The image to be padded.
            size (`Dict[str, int]`):
                The size `{"height": h, "width": w}` to pad the image to.
            random_padding (`bool`, *optional*, defaults to `False`):
                Whether to use random padding or not.
            data_format (`str` or `ChannelDimension`, *optional*):
                The data format of the output image. If unset, the same format as the input image is used.
        """
        output_height, output_width = size["height"], size["width"]
        input_height, input_width = get_image_size(image)

        delta_width = output_width - input_width
        delta_height = output_height - input_height

        if random_padding:
            pad_top = np.random.randint(low=0, high=delta_height + 1)
            pad_left = np.random.randint(low=0, high=delta_width + 1)
        else:
            pad_top = delta_height // 2
            pad_left = delta_width // 2

        pad_bottom = delta_height - pad_top
        pad_right = delta_width - pad_left

        bboxes_new = []
        if bboxes:
            for bbox in bboxes:
                bbox_new = [bbox[0]+pad_left, bbox[1]+pad_top, bbox[2]+pad_left, bbox[3]+pad_top]
                bboxes_new.append(bbox_new)

        padding = ((pad_top, pad_bottom), (pad_left, pad_right))
        return [pad(image, padding, data_format=data_format)], bboxes_new

    def pad(self, *args, **kwargs):
        logger.info("pad is deprecated and will be removed in version 4.27. Please use pad_image instead.")
        return self.pad_image(*args, **kwargs)

    def thumbnail_(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        bboxes = None,
        **kwargs,
    ):
        """
        Resize the image to make a thumbnail. The image is resized so that no dimension is larger than any
        corresponding dimension of the specified size.

        Args:
            image (`np.ndarray`):
                The image to be resized.
            size (`Dict[str, int]`):
                The size `{"height": h, "width": w}` to resize the image to.
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`):
                The resampling filter to use.
            data_format (`Optional[Union[str, ChannelDimension]]`, *optional*):
                The data format of the output image. If unset, the same format as the input image is used.
        """
        input_height, input_width = get_image_size(image)
        output_height, output_width = size["height"], size["width"]

        # We always resize to the smallest of either the input or output size.
        height = min(input_height, output_height)
        width = min(input_width, output_width)

        if height == input_height and width == input_width:
            return [image], bboxes

        if input_height > input_width:
            width = int(input_width * height / input_height)
        elif input_width > input_height:
            height = int(input_height * width / input_width)

        resized_image = resize(
            image, size=(height, width), resample=resample, reducing_gap=2.0, data_format=data_format, **kwargs
        )
        bboxes_new = []
        if bboxes:
            resize_rate = ( float(height) / image.shape[0] + float(width) / image.shape[1]) / 2
            for bbox in bboxes:
                bbox_new = [int(resize_rate * i) for i in bbox]
                bboxes_new.append(bbox_new)

        return [resized_image], bboxes_new

    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        bboxes = None,
        **kwargs,
    ):
        """
        Resize an image. The shortest edge of the image is resized to size["shortest_edge"], with the longest edge
        resized to keep the input aspect ratio.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Dict[str, int]`):
                Size of the output image.
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`):
                Resampling filter to use when resiizing the image.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
        """
        size = get_size_dict(size)
        shortest_edge = min(size["height"], size["width"])
        output_size = get_resize_output_image_size(image, size=shortest_edge, default_to_square=False)
        resized_image = resize(image, size=output_size, resample=resample, data_format=data_format, **kwargs)
        bboxes_new = []
        if bboxes:
            resize_rate = ( float(output_size[0]) / image.shape[0] + float(output_size[1]) / image.shape[1]) / 2
            for bbox in bboxes:
                bbox_new = [int(resize_rate * i) for i in bbox]
                bboxes_new.append(bbox_new)

        return [resized_image], bboxes_new

    def rescale(
        self,
        image: np.ndarray,
        scale: Union[int, float],
        data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ):
        """
        Rescale an image by a scale factor. image = image * scale.

        Args:
            image (`np.ndarray`):
                Image to rescale.
            scale (`int` or `float`):
                Scale to apply to the image.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
        """
        return rescale(image, scale=scale, data_format=data_format, **kwargs)

    def normalize(
        self,
        image: np.ndarray,
        mean: Union[float, List[float]],
        std: Union[float, List[float]],
        data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Normalize an image. image = (image - image_mean) / image_std.

        Args:
            image (`np.ndarray`):
                Image to normalize.
            image_mean (`float` or `List[float]`):
                Image mean.
            image_std (`float` or `List[float]`):
                Image standard deviation.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
        """
        return normalize(image, mean=mean, std=std, data_format=data_format, **kwargs)

    def preprocess(
        self,
        images: ImageInput,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        resample: PILImageResampling = None,
        do_thumbnail: bool = None,
        do_align_long_axis: bool = None,
        do_pad: bool = None,
        random_padding: bool = False,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        **kwargs,
    ) -> PIL.Image.Image:
    
        do_resize = do_resize if do_resize is not None else self.do_resize
        size = size if size is not None else self.size
        if isinstance(size, (tuple, list)):
            # Previous feature extractor had size in (width, height) format
            size = size[::-1]
        size = get_size_dict(size)
        resample = resample if resample is not None else self.resample
        do_thumbnail = do_thumbnail if do_thumbnail is not None else self.do_thumbnail
        do_align_long_axis = do_align_long_axis if do_align_long_axis is not None else self.do_align_long_axis
        do_pad = do_pad if do_pad is not None else self.do_pad
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std

        images = make_list_of_images(images)
        bboxes = kwargs['bboxes'] if 'bboxes' in kwargs.keys() else None 

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        if do_resize and size is None:
            raise ValueError("Size must be specified if do_resize is True.")

        if do_rescale and rescale_factor is None:
            raise ValueError("Rescale factor must be specified if do_rescale is True.")

        if do_pad and size is None:
            raise ValueError("Size must be specified if do_pad is True.")

        if do_normalize and (image_mean is None or image_std is None):
            raise ValueError("Image mean and std must be specified if do_normalize is True.")

        # All transformations expect numpy arrays.
        images = [to_numpy_array(image) for image in images]

        if do_align_long_axis:
            images = [self.align_long_axis(image, size=size) for image in images]

        if do_resize:
            images, bboxes = self.resize(image=images[0], size=size, resample=resample, bboxes=bboxes)
            # images, bboxes = [self.resize(image=image, size=size, resample=resample, bboxes=bboxes) for image in images]

        if do_thumbnail:
            try:
                images, bboxes = self.thumbnail_(image=images[0], size=size, bboxes=bboxes)
            except:
                res = self.thumbnail_(image=images[0], size=size, bboxes=bboxes)
                print(res)
                print(images.shape)
                print(bboxes.shape)
                import sys
                sys.exit()


        if do_pad:
            images, bboxes = self.pad_image(image=images[0], size=size, random_padding=random_padding, bboxes=bboxes)

        if do_rescale:
            images = [self.rescale(image=image, scale=rescale_factor) for image in images]

        if do_normalize:
            images = [self.normalize(image=image, mean=image_mean, std=image_std) for image in images]

        images = [to_channel_dimension_format(image, data_format) for image in images]

        data = {
            "pixel_values": images,
            "bboxes": bboxes
            }
        return BatchFeature(data=data, tensor_type=return_tensors)

    def preprocess_test(
        self,
        images: ImageInput,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        resample: PILImageResampling = None,
        do_thumbnail: bool = None,
        do_align_long_axis: bool = None,
        do_pad: bool = None,
        random_padding: bool = False,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        **kwargs,
    ) -> PIL.Image.Image:
        do_resize = do_resize if do_resize is not None else self.do_resize
        size = size if size is not None else self.size
        if isinstance(size, (tuple, list)):
            # Previous feature extractor had size in (width, height) format
            size = size[::-1]
        size = get_size_dict(size)
        resample = resample if resample is not None else self.resample
        do_thumbnail = do_thumbnail if do_thumbnail is not None else self.do_thumbnail
        do_align_long_axis = do_align_long_axis if do_align_long_axis is not None else self.do_align_long_axis
        do_pad = do_pad if do_pad is not None else self.do_pad
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std

        images = make_list_of_images(images)
        bboxes = kwargs['bboxes'] if 'bboxes' in kwargs.keys() else None 

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        if do_resize and size is None:
            raise ValueError("Size must be specified if do_resize is True.")

        if do_rescale and rescale_factor is None:
            raise ValueError("Rescale factor must be specified if do_rescale is True.")

        if do_pad and size is None:
            raise ValueError("Size must be specified if do_pad is True.")

        if do_normalize and (image_mean is None or image_std is None):
            raise ValueError("Image mean and std must be specified if do_normalize is True.")

        # All transformations expect numpy arrays.
        images = [to_numpy_array(image) for image in images]

        if do_align_long_axis:
            images = [self.align_long_axis(image, size=size) for image in images]

        if do_resize:
            images, bboxes = self.resize(image=images[0], size=size, resample=resample, bboxes=bboxes)
            # images, bboxes = [self.resize(image=image, size=size, resample=resample, bboxes=bboxes) for image in images]

        if do_thumbnail:
            try:
                images, bboxes = self.thumbnail_(image=images[0], size=size, bboxes=bboxes)
            except:
                res = self.thumbnail_(image=images[0], size=size, bboxes=bboxes)
                print(res)
                print(images.shape)
                print(bboxes.shape)
                import sys
                sys.exit()


        if do_pad:
            images, bboxes = self.pad_image(image=images[0], size=size, random_padding=random_padding, bboxes=bboxes)
        images = [to_channel_dimension_format(image, data_format) for image in images]
        data = {
            "pixel_values": images,
            "bboxes": bboxes
            }
        return data

