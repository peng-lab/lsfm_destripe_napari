import numpy as np
from aicsimageio.writers import OmeTiffWriter

from destripe_lsfm import napari_get_reader


# tmp_path is a pytest fixture
def test_reader(tmp_path):
    """An example of how you might test your plugin."""

    # write some fake data using your supported file format
    my_test_file = str(tmp_path / "myfile.tiff")
    original_data = np.random.randint(0, 2**16, (20, 20), dtype=np.uint16)
    data = original_data.astype(np.uint16)
    OmeTiffWriter.save(data, my_test_file, dim_order_out="YX")

    # try to read it back in
    reader = napari_get_reader(my_test_file)
    assert callable(reader)

    # make sure we're delivering the right format
    result = reader(my_test_file)
    assert isinstance(result, tuple) and len(result) > 0
    image, filename = result
    assert isinstance(image, np.ndarray)
    assert filename == "myfile.tiff"

    # make sure it's the same as it started
    assert image.shape == original_data.shape
    assert image.dtype == original_data.dtype


def test_get_reader_pass():
    reader = napari_get_reader("fake.file")
    assert reader is None
