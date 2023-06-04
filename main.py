from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch
from PIL import Image
from torchvision import transforms

from src.net import Encoder, Decoder

transform = transforms.ToTensor()


def callback_encode(args):
    model = Encoder()
    weights = torch.load(args.weights_path, map_location="cpu")
    model.load_state_dict(weights)
    model.to(args.device)
    model.eval()

    image = Image.open(args.input_path)
    input = transform(image)[None].to(args.device)
    output = model(input).cpu()
    torch.save(output, args.output_path)


def callback_decode(args):
    model = Decoder()
    weights = torch.load(args.weights_path, map_location="cpu")
    model.load_state_dict(weights)
    model.to(args.device)
    model.eval()

    input = torch.load(args.input_path).to(args.device)
    output = model(input).cpu().numpy()
    image = Image.fromarray(output)
    image.save(args.output_path)


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
        help="path to encoded image",
    )
    encoder_parser.set_defaults(callback=callback_encode)

    decoder_parser = subparsers.add_parser(
        "decode",
        help="decode input image",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    encoder_parser.add_argument(
        "-m",
        "--weights-path",
        help="path to model weights",
    )
    decoder_parser.add_argument(
        "-src",
        "--input-path",
        help="path to the image to encode",
    )
    decoder_parser.add_argument(
        "-dst",
        "--output-path",
        help="path to decoded image",
    )
    decoder_parser.set_defaults(callback=callback_decode)


def main():
    """Main function for forecasting asset revenue"""
    parser = ArgumentParser(
        prog="web spy",
        description="tool to spy for gitlab features",
    )
    setup_parser(parser)
    arguments = parser.parse_args()
    arguments.callback(arguments)


if __name__ == "__main__":
    main()
