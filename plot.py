import matplotlib.pyplot as plt
import argparse
import json

def plot_3d(data, ax, name):
    x = [i[0] for i in data]
    y = [i[1] for i in data]
    z = [i[2] for i in data]
    ax.plot_trisurf(x, y, z, label=name)

def main():
    parser = argparse.ArgumentParser(description='Plot miss rate of different loop orders')
    parser.add_argument('--input', type=str, default=None, help='input file')
    args = parser.parse_args()
    data = {}
    if args.input:
        with open(args.input, 'r') as f:
            data = json.load(f)
    else:
        allinput = input()
        data = json.loads(allinput)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for name, results in data.items():
        plot_3d(results, ax, name)
    ax.set_xlabel('M')
    ax.set_ylabel('N')
    ax.set_zlabel('Miss rate')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()