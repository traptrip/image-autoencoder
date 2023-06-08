from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from src.net import Encoder, Decoder
from src.quantization_utils import Compressor

MEAN = 0.485, 0.456, 0.406
STD = 0.229, 0.224, 0.225
to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
to_pil = transforms.ToPILImage()


@torch.inference_mode()
def callback_encode(args):
    model = Encoder()
    weights = torch.load(args.weights_path, map_location="cpu")["encoder"]
    model.load_state_dict(weights)
    model.to(args.device)
    model.eval()

    image = Image.open(args.input_path)
    input = to_tensor(image)[None].to(args.device)
    output = model(input).cpu()

    compressor = Compressor(args.q_level)
    q_embedding, shape = compressor.compress(output)
    np.savez_compressed(args.output_path, emb=q_embedding, shape=shape)


@torch.inference_mode()
def callback_decode(args):
    model = Decoder()
    weights = torch.load(args.weights_path, map_location="cpu")["decoder"]
    model.load_state_dict(weights)
    model.to(args.device)
    model.eval()

    compressor = Compressor(args.q_level)
    inputs = np.load(args.input_path)
    embedding = compressor.decompress(**inputs)
    embedding = embedding.to(args.device)

    output = model(embedding).cpu().numpy()[0]
    output = output.transpose(1, 2, 0)
    output = (output * 255 / output.max()).astype(np.uint8)
    image = Image.fromarray(output)
    image.save(args.output_path)


def setup_parser(parser: ArgumentParser):
    """Setup arguments parser for CLI"""
    parser.add_argument("-d", "--device", help="device to execute model", default="cpu")
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
