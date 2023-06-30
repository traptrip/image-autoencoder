from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from src.net import Encoder, Decoder
from src.quantization_utils import Compressor

MEAN = 0.485, 0.456, 0.406
STD = 0.229, 0.224, 0.225
# MEAN = 0.5, 0.5, 0.5
# STD = 0.5, 0.5, 0.5

to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
to_pil = transforms.ToPILImage()
# encoder = Encoder({"name": "src.encoders.resnet.ResNet18Encoder", "args": None})
# decoder = Decoder(
#     {"name": "src.decoders.resnet.ResNet18Decoder", "args": {"in_channels": 512}}
# )
encoder = Encoder({"name": "src.encoders.basic.Encoder", "args": None})
decoder = Decoder({"name": "src.decoders.basic.BasicDecoder", "args": None})
# encoder = Encoder({"name": "src.encoders.vit.Encoder", "args": None})
# decoder = Decoder({"name": "src.decoders.basic.ViTDecoder", "args": None})


@torch.inference_mode()
def encode(
    weights_path: str, input_path: str, output_path: str, q_level: int, device: str
):
    weights = torch.load(weights_path, map_location="cpu")["encoder"]
    encoder.load_state_dict(weights)
    encoder.to(device)
    encoder.eval()

    image = Image.open(input_path)
    input = to_tensor(image)[None].to(device)
    output = encoder(input).cpu()

    # np.savez_compressed(output_path, emb=output.numpy(), shape=output.numpy().shape)

    compressor = Compressor(q_level)
    q_embedding, shape = compressor.compress(output)
    np.savez_compressed(output_path, emb=q_embedding, shape=shape)


def callback_encode(args):
    encode(
        args.weights_path, args.input_path, args.output_path, args.q_level, args.device
    )


@torch.inference_mode()
def decode(
    weights_path: str, input_path: str, output_path: str, q_level: int, device: str
):
    weights = torch.load(weights_path, map_location="cpu")["decoder"]
    decoder.load_state_dict(weights)
    decoder.to(device)
    decoder.eval()

    compressor = Compressor(q_level)
    inputs = np.load(input_path)

    embedding = compressor.decompress(**inputs)
    # embedding = torch.tensor(inputs["emb"])

    embedding = embedding.to(device)

    output = decoder(embedding).cpu().numpy()[0]
    output = output.transpose(1, 2, 0)
    output = (output * 255).astype(np.uint8)
    image = Image.fromarray(output)
    image.save(output_path)


def callback_decode(args):
    decode(
        args.weights_path, args.input_path, args.output_path, args.q_level, args.device
    )


def setup_parser(parser: ArgumentParser):
    """Setup arguments parser for CLI"""
    subparsers = parser.add_subparsers(help="Choose command")

    encoder_parser = subparsers.add_parser(
        "encode",
        help="encode input image",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    encoder_parser.add_argument(
        "-m",
        "--weights-path",
        help="path to model weights",
    )
    encoder_parser.add_argument(
        "-src",
        "--input-path",
        help="path to the image to encode",
    )
    encoder_parser.add_argument(
        "-dst",
        "--output-path",
        default="emb.npz",
        help="path to encoded image",
    )
    encoder_parser.add_argument(
        "-q",
        "--q-level",
        help="level of quantization (bigger = softer quantization)",
        type=int,
        default=2,
    )
    encoder_parser.add_argument(
        "-d", "--device", help="device to execute model", default="cpu"
    )
    encoder_parser.set_defaults(callback=callback_encode)

    decoder_parser = subparsers.add_parser(
        "decode",
        help="decode input image",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    decoder_parser.add_argument(
        "-m",
        "--weights-path",
        help="path to model weights",
    )
    decoder_parser.add_argument(
        "-src",
        "--input-path",
        default="emb.npz",
        help="path to the image to encode",
    )
    decoder_parser.add_argument(
        "-dst",
        "--output-path",
        help="path to decoded image",
    )
    decoder_parser.add_argument(
        "-q",
        "--q-level",
        help="level of quantization (bigger = softer quantization)",
        type=int,
        default=2,
    )
    decoder_parser.add_argument(
        "-d", "--device", help="device to execute model", default="cpu"
    )
    decoder_parser.set_defaults(callback=callback_decode)


def main():
    """Main function for forecasting asset revenue"""
    parser = ArgumentParser(
        prog="Image compression tool",
        description="tool to compress & decompress image usin autoencoder",
    )
    setup_parser(parser)
    arguments = parser.parse_args()
    arguments.callback(arguments)


if __name__ == "__main__":
    main()
