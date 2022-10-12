# Barcode imports
import base64
from io import BytesIO
import numpy as np
from PIL import Image

def getBarcode(json_barcode, width=1024, height=12, as_list=False):
    """Return barcode from the DB's video's JSON array."""
    if not json_barcode:
        return None
    np_array = np.array(json_barcode, dtype="int")  # Parse the numpy list from string
    flipped_array = np.flip(np.array(np_array).astype('int'), 1)  # Switch from BGR to RGB values
    new_img = np.expand_dims(np.array(flipped_array), axis=0)  # Add dimension to convert matrix to image tensor
    new_image = np.repeat(new_img, [224], axis=0)
    new_image = new_image.astype(np.uint8)
    img = Image.fromarray(new_image)  # Convert image tensor to Pillow Image object
    img = img.resize(size=(width, height))
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    if as_list:
        return np.array(img).tolist()
    else:
        base64_encoded_result_bytes = base64.b64encode(buffered.getvalue())
        base64_encoded_result_str = base64_encoded_result_bytes.decode('ascii')
        return base64_encoded_result_str