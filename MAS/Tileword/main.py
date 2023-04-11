from my_test import test_reaction, test_gamma, test_p, test_boldness, test_disappear
from init import init_args, reset_args, run

if __name__ == "__main__":
    args = init_args()
    # run(args)
    test_gamma(args)
    test_p(args)
    test_boldness(args)
    test_reaction(args)
    test_disappear(args)
