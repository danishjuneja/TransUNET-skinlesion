"""Build lightweight variants and verify output tensor shapes."""

from skin_lesion_segmentation.models import build_transunet, build_unet


def main():
    unet = build_unet(image_size=64, base_filters=8)
    transunet = build_transunet(
        image_size=64,
        hidden_size=64,
        transformer_layers=1,
        heads=4,
        mlp_dim=128,
        decoder_channels=(64, 32, 16, 8),
        pretrained=False,
    )
    assert unet.output_shape == (None, 64, 64, 1)
    assert transunet.output_shape == (None, 64, 64, 1)
    print(f"U-Net parameters: {unet.count_params():,}")
    print(f"TransUNet parameters: {transunet.count_params():,}")


if __name__ == "__main__":
    main()
