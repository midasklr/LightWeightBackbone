# LightWeightBackbone

Light weight backbone trained with imagenet.

- [x] mobilenetv3 small 0.1
- [x] mobilenetv3 small 0.75
- [ ] mobilenetv2 0.25
- [ ] ..

## imagenet val results

| model                        | hyp           | top1   | top5   |
| ---------------------------- | ------------- | ------ | ------ |
| mobilenetv3 small 0.1        | adamw;bs 2048 | 36.128 | 60.336 |
| mobilenetv3 small 0.75-paper | -             | 65.4   | -      |
| mobilenetv3 small 0.75       | adamw;bs 1024 | 64.424 | 85.386 |
| mobilenetv3 small 0.5        |               | 55.94  | 78.682 |

## Train

```
python main.py --dist-url 'tcp://127.0.0.1:12342' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 --gpu 0123 data -b 1024
```

## Demo

```
python demo.py
```

