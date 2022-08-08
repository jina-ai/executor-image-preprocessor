from typing import Optional, Tuple, Any, Dict

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
        traversal_paths: str = "@r",
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
        self.traversal_paths = traversal_paths

    @staticmethod
    def _get_channel_axis(doc: Document) -> Optional[int]:
        """Find the channel axis in an image tensor."""
        if hasattr(doc, "tensor"):
            # print(doc.tensor.shape)
            for axis, dim in enumerate(doc.tensor.shape):
                if dim == 3:
                    return axis
            raise ValueError(f'Could not find channel axis in document with id: {doc.id}')

    def _reshape(self, docs: DocumentArray):
        """Reshape images."""
        for doc in docs:
            if hasattr(doc, "tensor") and doc.tensor is not None:
                channel_axis = self._get_channel_axis(doc)
                if channel_axis != self._channel_axis:
                    doc.set_image_tensor_channel_axis(channel_axis, self._channel_axis)
                doc.set_image_tensor_shape(self._shape, self._channel_axis)

    @staticmethod
    def _normalize(docs: DocumentArray) -> None:
        """Normalize images."""
        for doc in docs:
            if hasattr(doc, "tensor") and doc.tensor is not None:
                doc.tensor = (doc.tensor / 127.5) - 1

    @requests
    def preprocess(self, docs: DocumentArray, parameters: Dict[str, Any], **_) -> Optional[DocumentArray]:
        """Preprocess docs."""
        traversal_paths = parameters.get("traversal_paths", self.traversal_paths)
        img_extensions = ["png", "jpg", "jpeg", "gif"]
        for d in docs[traversal_paths]:
            if d.uri and d.tensor is None:
                d.load_uri_to_image_tensor()

            # if not d.text:
                # print(d.content, "no text")

                # if d.tensor is not None and not d.text:
            # if hasattr(d, "tensor"):
        self._reshape(docs[traversal_paths])
        self._normalize(docs[traversal_paths])
        return docs
