from typing import Optional, Tuple

from jina import requests, Document, DocumentArray, Executor


class ImagePreprocessor(Executor):
    """
    An executor that can be used for standard image pre-processing, i.e. resizing
    and normalization.
    """

    def __init__(
        self,
        shape: Tuple[int, int] = (200, 200),
        channel_axis: int = 0,
        **kwargs,
    ) -> None:
        """
        Initialization

        :param shape: The output image shape.
        :param channel_axis: The output channel axis.
        """
        super().__init__(**kwargs)
        assert len(shape) == 2
        assert 0 <= channel_axis <= 2
        self._shape = shape
        self._channel_axis = channel_axis

    @staticmethod
    def _get_channel_axis(doc: Document) -> Optional[int]:
        """Find the channel axis in an image blob."""
        for axis, dim in enumerate(doc.blob.shape):
            if dim == 3:
                return axis
        raise ValueError(f'Could not find channel axis in document with id: {doc.id}')

    def _reshape(self, docs: DocumentArray):
        """Reshape images."""
        for doc in docs:
            channel_axis = self._get_channel_axis(doc)
            if channel_axis != self._channel_axis:
                doc.set_image_blob_channel_axis(channel_axis, self._channel_axis)
            doc.set_image_blob_shape(self._shape, self._channel_axis)

    @staticmethod
    def _normalize(docs: DocumentArray) -> None:
        """Normalize images."""
        docs.blobs = (docs.blobs / 127.5) - 1

    @requests
    def preprocess(self, docs: DocumentArray, **_) -> Optional[DocumentArray]:
        """Preprocess docs."""
        for d in docs:
            if d.uri and not d.blob:
                d.load_uri_to_image_blob()
        self._reshape(docs)
        self._normalize(docs)
        return docs
