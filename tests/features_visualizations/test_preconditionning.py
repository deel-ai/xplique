from xplique.features_visualizations.preconditioning import fft_image, \
    fft_to_rgb, get_fft_scale

def test_shape():
    """Ensure the shape of the image generated is correct"""

    input_shapes = [(2, 8, 8, 3), (2, 64, 64, 1), (2, 256, 256, 3), (2, 512, 512, 1)]

    for input_shape in input_shapes:
        n, w, h, c = input_shape
        buffer = fft_image(input_shape)
        scaler = get_fft_scale(w, h)
        img = fft_to_rgb(input_shape, buffer, scaler)

        # image generated from inverse fourier transformation should have
        # the desired shape
        assert img.shape == input_shape
