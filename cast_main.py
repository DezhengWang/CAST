import argparse
import os.path
import torch
from exp.exp_basic import Exp_Basic
from tools import log, parser_args
from models.CAST.cast import CAST

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='[CAST] Long Sequences Forecasting')

    ## Datasets Related Hyper-Parameters
    parser.add_argument('--root_path', type=str, default='data', help='root path of dataset file')
    parser.add_argument('--data_path', type=str, default="ETTh1.csv", help='file name of dataset file')
    parser.add_argument('--target', type=int, default=1, help='index of target feature, started from 0')
    parser.add_argument('--ration', type=list, default=[], help='length of train, valid, and test datasets')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input feature size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input feature size')
    parser.add_argument('--c_out', type=int, default=7, help='output feature size')

    ## Dataloader Related Hyper-Parameters
    parser.add_argument('--size', type=list, default=[96, 24, 24], help='length of sequence, label, and prediction')
    parser.add_argument('--iter_horizon', type=int, default=12, help='horizon of iteration for prediction')
    parser.add_argument('--features', type=str, default="M",
                        help='forecasting task, options:[M, MS, S]; M:multivariate, MS: multivariate to univariate, S:univariate')
    parser.add_argument('--scale', type=bool, default=True, help='StandardScaler')
    parser.add_argument('--inverse', type=bool, default=False, help='Inverse StandardScaler')
    parser.add_argument('--aug', default=None, help='data augmentation')
    parser.add_argument('--maskrate', type=float, default=0.2, help='mask rate of randmask')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')

    ## CAST Related Hyper-Parameters
    parser.add_argument('--net', default=CAST, help='Net Structure')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--nhead', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=2, help='num of decoder layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--initial_learning_rate', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of results')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--epochs', type=int, default=20, help='train epochs')
    parser.add_argument('--step_size', type=int, default=2, help='adjust lr a step size')

    ## Interpreter Related Hyper-Parameters
    parser.add_argument('--cuda', type=str, default=0, help='device ids of gpus')
    parser.add_argument('--iteration', type=bool, default=True, help='prediction via iteration')

    args = parser.parse_args([])
    iter_nums = range(1, 11)
    for args.iteration in [False]:
        for iter_num in iter_nums:
            for args.aug in [None]:
                for args.root_path in ["ETTh1"]:
                    args = parser_args(args)
                    for args.size[2] in args.pre_lens:  # 24, 48, 96, 192, 336, 480, 624, 720
                        # CAST
                        for args.net in [CAST]: #
                            # init hyper parameters
                            args.learning_rate = args.initial_learning_rate
                            if not args.iteration:
                                args.iter_horizon = args.size[2]

                            aug = "None" if args.aug is None else '_'.join(
                                [aug_item.__name__ + f"({args.maskrate})" if aug_item.__name__ == "randmask" else aug_item.__name__
                                 for aug_item in args.aug])

                            args.checkpoints = os.path.join("checkpoints_ablation_v2", f"{args.root_path}",
                                                            f"{args.net.__name__}_Enc({args.e_layers})_Dec({args.d_layers})",
                                                            f"EncLen({args.size[0]})_DecLen({args.size[1]})_OutLen({args.size[2]})",
                                                            f"Iteration" if args.iteration else f"Once",
                                                            f"Aug_{aug}",
                                                            f"Repeat{iter_num}")

                            log(f"Training {args.net.__name__} on {args.root_path} with"
                                f" Enc: {args.e_layers}, Dec: {args.d_layers}, Heads: {args.nhead},"
                                f" via: {'Iteration' if args.iteration else 'Once'}, repeat {iter_num}/{iter_nums[-1]}",
                                level="Debug")
                            log(f"sequence length: {args.size[0]}," +
                                f" label length: {args.size[1]}," +
                                f" prediction length: {args.size[2]}," +
                                f" Augmentation: {aug}",
                                level="Debug")
                            # release cuda resource
                            torch.cuda.empty_cache()
                            # delete all related information
                            try:
                                del exp
                            except:
                                pass
                            #
                            try:
                                # initial
                                exp = Exp_Basic(args)
                                # load dataloader
                                train_loader, valid_loader, test_loader = exp._get_data()
                                # training model
                                exp.train_loop(train_loader, valid_loader)
                                # test CAST
                                exp.test(test_loader, load=True)
                            except Exception as e:
                                log(e, level="error")
                            # release cuda resource
                            torch.cuda.empty_cache()
    log("Training is Done!")
