# Real-ESRGAN

Оригинальный README может быть найден [здесь](https://github.com/xinntao/Real-ESRGAN).

## 🔧 Системные требования

- Python = 3.7
- [PyTorch >= 1.7](https://pytorch.org/)

## Установка

```bash
pip install basicsr
pip install facexlib
pip install gfpgan
pip install -r requirements.txt
python setup.py develop
```
---

## Использование

Загрузить веса: [RealESRGAN_x4plus.pth](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth)
```bash
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P weights
```
Инференс:

```console
Использование: `python inference_realesrgan.py -n RealESRGAN_x4plus -i infile -o outfile [options]...`

Типичная команда: python inference_realesrgan.py -n RealESRGAN_x4plus -i infile --outscale 3.5 --face_enhance

  -h                   Помощь
  -i --input           Директория с изображениями для инференса. По умолчанию: inputs
  -o --output          Директория вывода. По умолчанию: results
  -n --model_name      Имя модели. По умолчанию: RealESRGAN_x4plus
  -s, --outscale       Коэффициент масштабирования. По умолчанию: 4
  --suffix             Суффикс масштабированного изображения. По умолчанию: out
  -t, --tile           Размер тайла, 0 - без тайлов. По умолчанию: 0
  --face_enhance       Использовать GFPGAN для улучшения лиц. По умолчанию: False
  --fp32               Использовать тип fp32 для инференса. По умолчанию: fp16 (half precision).
  --ext                Формат изображения. Опции: auto | jpg | png, auto значит использовать то же разрешение, что и у входного изображения. По умолчанию: auto
  --pre_downscale      Уменьшить масштаб входного изображения перед инференсом на заданный коэффициент.
```

```bash
python inference_realesrgan.py -n RealESRGAN_x4plus -i inputs --face_enhance
```
Results are in the `results` folder

Примеры:
```
python inference_realesrgan.py -n RealESRGAN_x4plus -i ../datasets/DIV2K/DIV2K -o results/DIV2K --outscale 2 --pre_downscale 2
python inference_realesrgan.py -n RealESRGAN_x4plus -i ../datasets/General100/General100 -o results/General100 --outscale 2 --pre_downscale 2
python inference_realesrgan.py -n RealESRGAN_x4plus -i ../datasets/BSDS100/BSDS100 -o results/BSDS100 --outscale 2 --pre_downscale 2
python inference_realesrgan.py -n RealESRGAN_x4plus -i ../datasets/urban100/urban100 -o results/urban100 --outscale 2 --pre_downscale 2
python inference_realesrgan.py -n RealESRGAN_x4plus -i ../datasets/set14/set14/set14 -o results/set14 --outscale 2 --pre_downscale 2
```