from jina import Executor, DocumentArray, requests
import random
import os


def generate_price(price_range=(10, 200)):
    price = random.randrange(price_range[0], price_range[1])

    return price

def generate_uri(doc, data_dir=".", file_ext="jpg"):
    doc.uri = f"{data_dir}/{doc.id}.{file_ext}"

    return doc


def preproc(doc, tensor_shape=(80, 60)):
    # ensure we have a tensor
    if doc.uri:
        doc.load_uri_to_image_tensor()
    elif doc.blob:
        doc.convert_blob_to_image_tensor()

    # Apply settings to tensor
    doc.set_image_tensor_shape(tensor_shape).set_image_tensor_normalization()

    return doc


def add_metadata(doc, data_dir="./data", rating_range=(0, 5), price_range=(10, 200)):
    # Fix uri
    if not data_dir:
        data_dir = "."

    if hasattr(doc, "id"):
        filename = f"{data_dir}/{doc.id}.jpg"
        doc.uri = filename

    # Generate fake price
    if price_range:
        doc.tags["price"] = generate_price(price_range)

    # Generate fake rating based on id
    if rating_range:
        random.seed(int(doc.id))  # Ensure reproducability
        doc.tags["rating"] = random.randrange(rating_range[0], rating_range[1])

    return doc


class FashionSearchPreprocessor(Executor):
    def __init__(
        self,
        data_dir="./data",
        tensor_shape=(80, 60),
        rating_range=(0, 5),
        price_range=(10, 200),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.data_dir = data_dir
        self.tensor_shape = tensor_shape
        self.price_range = price_range
        self.rating_range = rating_range

    @requests(on="/index")
    def process_index_document(self, docs: DocumentArray, **kwargs):
        for doc in docs:
            doc = generate_uri(doc, data_dir=self.data_dir)
            doc = preproc(doc)
            doc = add_metadata(doc, data_dir=self.data_dir)

    @requests(on="/search")
    def process_search_document(self, docs: DocumentArray, **kwargs):
        for doc in docs:
            doc = preproc(doc, "../data/images")
