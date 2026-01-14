from gawee_ir.parser import Parser
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path')
    args = parser.parse_args()
    g = Parser.parse_onnx(args.path)
    g.dump()
    return 


if __name__ == "__main__":
    main()