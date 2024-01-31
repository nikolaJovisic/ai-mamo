import json
import os

import numpy as np
from matplotlib import pyplot as plt
from pynetdicom import AE, debug_logger, evt
from pynetdicom.sop_class import \
    DigitalMammographyXRayImageStorageForPresentation
from ai.inference import infer

# Enable logging
debug_logger()


with open(f'config.json', 'r') as config_file:
    config_data = json.load(config_file)

# Define event handlers
def handle_store(event):
    """Handle a C-STORE request event."""
    ds = event.dataset
    ds.file_meta = event.file_meta

    # Save the dataset using a unique filename
    image_id = ds["AccessionNumber"].value
    os.mkdir(os.path.join(config_data['data_path'], image_id))

    filename = os.path.join(config_data['data_path'], image_id, f"{image_id}.dcm")
    ds.save_as(filename, write_like_original=False)
    print(f"Stored DICOM file: {filename}")

    image = extract_image(ds)
    mask = infer(image, binarize=False)

    plt.imsave(os.path.join(config_data['data_path'], image_id, f"{image_id}.png"), mask, cmap='gray')

    # Return a 'Success' status
    return 0x0000


def extract_image(ds):
    pixel_data = ds.PixelData
    pixel_array = np.frombuffer(pixel_data, dtype=np.uint16)
    rows = ds.Rows
    columns = ds.Columns
    image = pixel_array.reshape((rows, columns))
    return image


handlers = [(evt.EVT_C_STORE, handle_store)]

# Define the AE
ae = AE()

# Add the supported presentation context
ae.add_supported_context(DigitalMammographyXRayImageStorageForPresentation)

# Start the AE
ae.start_server(("0.0.0.0", 11112), block=True, evt_handlers=handlers)
