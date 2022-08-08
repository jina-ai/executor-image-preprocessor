from executor import ImagePreprocessor
from docarray import DocumentArray, Document

docs = DocumentArray(
    [
        Document(uri="idris.jpg"),
        Document(text="foo"),
    ]
)

print(docs.summary())
print([d.mime_type for d in docs])

exec = ImagePreprocessor()

exec.preprocess(docs, parameters={})


for doc in docs:
    print(doc)
