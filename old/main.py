import argparse, os, torch

from ACGAN import ACGAN


# parsing and configuration
def parse_args():
    args = argparse.Namespace(
        gan_type='ACGAN',
        dataset='mnist',
        epoch=50,
        batch_size=64,
        input_size=28,
        save_dir='models',
        result_dir='results',
        log_dir='logs',
        lrG=0.0002,
        lrD=0.0002,
        beta1=0.5,
        beta2=0.999,
        gpu_mode=True,
        benchmark_mode=True
    )
    return check_args(args)


# checking arguments
def check_args(args):
    # --save_dir
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # --result_dir
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    # --result_dir
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')

    return args


# main
def main():
    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    if args.benchmark_mode:
        torch.backends.cudnn.benchmark = True

    gan = ACGAN(args)

    import random
    random_range = [random.randint(0, 9) for _ in range(20)]
    flag = False
    for i in random_range:
        if not flag:
            flag = True
            res = gan.generate(i)
        else:
            res = torch.cat((res, gan.generate(i)), dim=0)

    print(res.shape)


if __name__ == '__main__':
    main()
