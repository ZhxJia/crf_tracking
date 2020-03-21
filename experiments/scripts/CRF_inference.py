from math import exp
import csv
import numpy as np
import factorgraph as fg

# MOT09 :frame 437-447 id:5
# # MOT09 :frame 502-525 id:33
# # MOT05 :frame 562-570 id:61

# root = '/content/drive/My Drive/workspace/tracking_wo_bnw/output/tracktor/MOT17/Tracktor++/'

root = '../../Tracktor++/'
dataset = 'MOT17-09-FRCNN.txt'


def get_center(info):
    x1 = eval(info[0]) - 1
    y1 = eval(info[1]) - 1
    w = eval(info[2]) - 1
    h = eval(info[3]) - 1
    return np.array([x1 + w / 2, y1 + h / 2])


def get_size(info):
    w = eval(info[2]) - 1
    h = eval(info[3]) - 1
    return (w + h) / 2


# [frame+1, i+1, x1+1, y1+1, x2-x1+1, y2-y1+1, str(torch.tensor(score)), is_fake]
with open(root + dataset, 'r') as f:
    track_dict = {}
    for line in f.readlines():
        items = [item for item in line.split(',')]
        if items[1] not in track_dict.keys():  # items[0] -> frame ;  items[1] -> id
            track_dict[items[1]] = {}
        track_dict[items[1]][items[0]] = items[2:]

all_info = {}
for track_id, id_info in track_dict.items():
    track_id = int(track_id) - 1
    tracks_count = 0
    id_frame_info_t_2 = []
    id_frame_info_t_1 = []
    last_id_frame = 0
    for id_frame, id_frame_info in id_info.items():
        id_frame = int(id_frame)
        if id_frame != (last_id_frame + 1):
            tracks_count = 0
            id_frame_info_t_2 = []
            id_frame_info_t_1 = []
        last_id_frame = id_frame
        tracks_count += 1
        if tracks_count < 3:
            if len(id_frame_info_t_1) != 0:
                id_frame_info_t_2 = id_frame_info_t_1
            id_frame_info_t_1 = id_frame_info
            continue
        # calculate the change of velocity from frame t-1 to frame t
        v_t = get_center(id_frame_info) - get_center(id_frame_info_t_1)
        v_t_1 = get_center(id_frame_info_t_1) - get_center(id_frame_info_t_2)
        dv = v_t - v_t_1

        # calculate the change of 'size change' from frame t-1 to frame t
        l_t = get_size(id_frame_info) / get_size(id_frame_info_t_1)
        l_t_1 = get_size(id_frame_info_t_1) / get_size(id_frame_info_t_2)
        dl = l_t / l_t_1

        # store the calculation results in the according frame in the dict 'all_info'
        if id_frame not in all_info.keys():
            all_info[id_frame] = []
        score = eval(id_frame_info[4].split(',')[0][-6:])
        all_info[id_frame].append([track_id, dv[0], dv[1], dl, score, int(id_frame_info[-1])])
        id_frame_info_t_2 = id_frame_info_t_1
        id_frame_info_t_1 = id_frame_info

# the larger the distance between Yi and Yj,
# the less we assume their labels to be the same

Wu = 0.8  # the weight of unary-type 0.8
Wb = 0.2  # the weight of binary-type 0.2


# calculate the potential of a unary-type factor
def calc_unary(F_info):  # F_info is a list
    score = F_info[4]
    if score > 0.99:
        beta = 20.
    else:
        beta = 5.
    Pu_0 = abs(0 - max(2 * score - 1, 0))
    Pu_1 = abs(1. - max(2. * score - 1., 0))

    return np.array([exp(-beta * Wu * Pu_0), exp(-beta * Wu * Pu_1)])
    # here we use parameter share, whereas we could also use different
    # parameters for each label configuration


def calc_binary(F_info):  # F_info is a list of two lists
    dvxi = F_info[0][1]
    dvxj = F_info[1][1]
    dvyi = F_info[0][2]
    dvyj = F_info[1][2]
    dli = F_info[0][3]
    dlj = F_info[1][3]
    alpha1 = 0.025
    alpha2 = 10
    k_y = 1.00  # 0.95
    k_x = 1.00  # 0.05
    # k_y = 1.0
    # k_x = 1.0
    Pb_00_11 = alpha1 * (k_x * (dvxi - dvxj) ** 2 + k_y * (dvyi - dvyj) ** 2) + alpha2 * abs(dli - dlj)
    Pb_00_11 = Wb * Pb_00_11
    return np.array([[exp(-Pb_00_11), exp(-0)], [exp(-0), exp(-Pb_00_11)]])


test_frame_id = 443
for test_frame_id in range(562, 570):
    frame_info = all_info[test_frame_id]

    g = fg.Graph()

    node_id_list = [l[0] for l in frame_info]

    # create random variable node for track_id
    for i in range(len(node_id_list)):
        node_i = str(node_id_list[i])  # node_i is the track_id (turned into str) for the variable node i
        g.rv(node_i, 2)  # yi can be 0 or 1

    for i in range(len(node_id_list)):
        node_i = str(node_id_list[i])
        # first, add the unary factors
        # frame_info[i] -> [id,dvx,dvy,dl,score,is_fake]
        g.factor([node_i], potential=calc_unary(frame_info[i]))  # Create unary factor with potential function
        # second, add the binary factors
        for j in range(len(node_id_list)):
            if j <= i:
                continue
            node_j = str(node_id_list[j])
            g.factor([node_i, node_j],
                     potential=calc_binary([frame_info[i], frame_info[j]]))  # Create binary with potential func

    # Run (loopy) belief propagation (LBP)
    iters, converged = g.lbp_MAP(normalize=True)
    print('LBP ran for %d iterations. Converged = %r' % (iters, converged))
    print('\n')
    print(f'frame id:{test_frame_id} in ' + dataset)
    # # Print out the final messages from LBP
    # g.print_messages()
    # print('\n')
    # Print out the final marginals
    # g.print_rv_marginals()
    g.print_rv_MAP()  # 推倒标签
