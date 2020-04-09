import os
from cycler import cycler as cy
from collections import defaultdict
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

colors = [
    'aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure', 'beige', 'bisque',
    'black', 'blanchedalmond', 'blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue',
    'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'cornsilk', 'crimson', 'cyan',
    'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkgrey', 'darkkhaki',
    'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon',
    'darkseagreen', 'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise',
    'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dimgrey', 'dodgerblue', 'firebrick',
    'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod',
    'gray', 'green', 'greenyellow', 'grey', 'honeydew', 'hotpink', 'indianred', 'indigo',
    'ivory', 'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue',
    'lightcoral', 'lightcyan', 'lightgoldenrodyellow', 'lightgray', 'lightgreen', 'lightgrey',
    'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray', 'lightslategrey',
    'lightsteelblue', 'lightyellow', 'lime', 'limegreen', 'linen', 'magenta', 'maroon',
    'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen',
    'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue',
    'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'navy', 'oldlace', 'olive', 'olivedrab',
    'orange', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise',
    'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue',
    'purple', 'rebeccapurple', 'red', 'rosybrown', 'royalblue', 'saddlebrown', 'salmon',
    'sandybrown', 'seagreen', 'seashell', 'sienna', 'silver', 'skyblue', 'slateblue',
    'slategray', 'slategrey', 'snow', 'springgreen', 'steelblue', 'tan', 'teal', 'thistle',
    'tomato', 'turquoise', 'violet', 'wheat', 'white', 'whitesmoke', 'yellow', 'yellowgreen'
]

def main():
    for seq_no in ['01','03','06','07','08','12','14']:
        txt_dir = "D:/Study/workspace/tracking_wo_bnw-master/test_results/public_CRF/plot_crf/" + "MOT17-" + seq_no + "-FRCNN.txt"
        im_dir = "D:/Study/workspace/tracking_wo_bnw-master/data/MOT17Det/test/" + "MOT17-" + seq_no + "/img1"
        output_dir="D:/Study/workspace/tracking_wo_bnw-master/output_vis/test/" + "MOT17-" + seq_no
        print(f'Plotting sequence: MOT17-{seq_no}')

        results={}
        with open(txt_dir, "r") as f:
            for line in f.readlines():
                items = line.split(',')
                frame = eval(items[0]) - 1
                index = eval(items[1]) - 1
                x1 = eval(items[2]) - 1
                y1 = eval(items[3]) - 1
                x2 = eval(items[4]) + x1 - 1
                y2 = eval(items[5]) + y1 - 1
                if not index in results:
                    results[index] = {}
                results[index][frame] = [x1, y1, x2, y2]
        plot_sequence(results,im_dir,output_dir)

def plot_sequence(tracks, im_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # infinite color loop
    cyl = cy('ec', colors)
    loop_cy_iter = cyl()
    styles = defaultdict(lambda: next(loop_cy_iter))

    im_names=os.listdir(im_dir)
    for i, im_name in enumerate(im_names):
        im_output = os.path.join(output_dir, im_name)
        im = Image.open(os.path.join(im_dir,im_name))

        sizes = np.shape(im)
        height = float(sizes[0])
        width = float(sizes[1])

        fig = plt.figure()
        fig.set_size_inches(width / 100, height / 100)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(im)

        for j, t in tracks.items():
            if i in t.keys():
                t_i = t[i]
                ax.add_patch(
                    plt.Rectangle(
                        (t_i[0], t_i[1]),
                        t_i[2] - t_i[0],
                        t_i[3] - t_i[1],
                        fill=False,
                        linewidth=1.0, **styles[j]
                    ))

                ax.annotate(j, (t_i[0] + (t_i[2] - t_i[0]) / 2.0, t_i[1] + (t_i[3] - t_i[1]) / 2.0),
                            color=styles[j]['ec'], weight='bold', fontsize=12, ha='center', va='center')

        plt.axis('off')
        # plt.tight_layout()
        plt.draw()
        plt.savefig(im_output, dpi=100)
        plt.close()

if __name__=="__main__":
    main()