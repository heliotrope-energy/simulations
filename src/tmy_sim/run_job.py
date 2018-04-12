#script intended for running with GridEngine

from argparse import ArgumentParser
from tmy import run
from sys import argv


def parse_args(args):
    parser = ArgumentParser()
    parser.add_argument("--tmy_file", help="location of TMY file")
    parser.add_argument("--albedo", help="albedo param", type=float)
    parser.add_argument("--steps", help="number of steps")
    parser.add_argument("--output", help="output directory")
    return parser.parse_args(args)


def run_job(args):
    print("running on data {}".format(args.tmy_file))
    output_num = args.tmy_file.split(".")[-1]
    run(args.tmy_file, args.albedo, args.output, str(output_num), steps=args.steps)


if __name__=="__main__":
    run_job(parse_args(argv[1:]))
