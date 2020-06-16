from KM_KBQA.qa.main_cors import run
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-sa', '--server_addr', required=True)
parser.add_argument('-sp', '--server_port', required=True)
parser.add_argument('-lp', '--local_port', required=True)

args = parser.parse_args()

if __name__ == '__main__':
    run(args)
