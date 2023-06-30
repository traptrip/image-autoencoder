# b=2
# echo 'B=2'
# echo 'Encoding...'
# python main.py encode -m runs/basic__mse_vgg__b=2/best_checkpoint.pth -src assets/images/baboon.png -dst assets/compressed/b=2/baboon.npz -q 2
# python main.py encode -m runs/basic__mse_vgg__b=2/best_checkpoint.pth -src assets/images/lena.png -dst assets/compressed/b=2/lena.npz -q 2
# python main.py encode -m runs/basic__mse_vgg__b=2/best_checkpoint.pth -src assets/images/peppers.png -dst assets/compressed/b=2/peppers.npz -q 2

# echo 'Decoding...'
# python main.py decode -m runs/basic__mse_vgg__b=2/best_checkpoint.pth -src assets/compressed/b=2/baboon.npz -dst assets/decompressed/b=2/baboon.png -q 2
# python main.py decode -m runs/basic__mse_vgg__b=2/best_checkpoint.pth -src assets/compressed/b=2/lena.npz -dst assets/decompressed/b=2/lena.png -q 2
# python main.py decode -m runs/basic__mse_vgg__b=2/best_checkpoint.pth -src assets/compressed/b=2/peppers.npz -dst assets/decompressed/b=2/peppers.png -q 2


echo 'B=8'
echo 'Encoding...'
python main.py encode -m runs/universal__basic_v2__mse/best_checkpoint.pth -src assets/images/baboon.png -dst assets/compressed/b=8/baboon.npz -q 8
python main.py encode -m runs/universal__basic_v2__mse/best_checkpoint.pth -src assets/images/lena.png -dst assets/compressed/b=8/lena.npz -q 8
python main.py encode -m runs/universal__basic_v2__mse/best_checkpoint.pth -src assets/images/peppers.png -dst assets/compressed/b=8/peppers.npz -q 8

echo 'Decoding...'
python main.py decode -m runs/universal__basic_v2__mse/best_checkpoint.pth -src assets/compressed/b=8/baboon.npz -dst assets/decompressed/b=8/baboon.png -q 8
python main.py decode -m runs/universal__basic_v2__mse/best_checkpoint.pth -src assets/compressed/b=8/lena.npz -dst assets/decompressed/b=8/lena.png -q 8
python main.py decode -m runs/universal__basic_v2__mse/best_checkpoint.pth -src assets/compressed/b=8/peppers.npz -dst assets/decompressed/b=8/peppers.png -q 8


# python main.py encode -m runs/basic__mse_vgg__b=2/best_checkpoint.pth -src assets/images/lena.png -dst assets/compressed/b=2/lena.npz -q 2
# python main.py decode -m runs/basic__mse_vgg__b=2/best_checkpoint.pth -src assets/compressed/b=2/lena.npz -dst assets/decompressed/b=2/lena.png -q 2
