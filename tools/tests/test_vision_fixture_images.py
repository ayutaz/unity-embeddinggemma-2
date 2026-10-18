import hashlib
import io

from PIL import Image

from embeddinggemma_tools.vision_fixture_images import save_vision_fixture


def test_known_procedural_pixels_reuse_exact_reviewed_png_bytes(tmp_path):
    image=Image.new('RGB',(96,96))
    image.putdata([(x*7%256,y*11%256,(x+y)*5%256) for y in range(96) for x in range(96)])
    output=tmp_path/'fixture.png'
    save_vision_fixture(image,output)
    assert hashlib.sha256(output.read_bytes()).hexdigest()=='c2a1900a1656ce2d843e9d06801c62f927b98a51e9b95ae1159bb2c7515fa399'
    with Image.open(output) as restored:
        assert restored.tobytes()==image.tobytes()


def test_different_input_pixels_are_preserved_instead_of_replaced(tmp_path):
    image=Image.new('RGB',(96,96),'red')
    expected=io.BytesIO();image.save(expected,format='PNG')
    output=tmp_path/'fixture.png'
    save_vision_fixture(image,output)
    assert output.read_bytes()==expected.getvalue()
