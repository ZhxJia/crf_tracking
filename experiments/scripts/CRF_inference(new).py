from math import exp
import csv
import numpy as np
import src.tracktor.factorgraph as fg

'''
 MOT09 :frame 437-447 id:5
 MOT09 :frame 502-525 id:33
 MOT05 :frame 562-570 id:61
 MOT11 :frame 798-817 id:53 reid的                                                                                      错误问题
'''

root = '../../Tracktor++/'
dataset = 'MOT17-09-FRCNN.txt'
frame = [214, 216]


def get_center(info):
    x1 = eval(info[0]) - 1
    y1 = eval(info[1]) - 1
    w = eval(info[2]) - 1
    h = eval(info[3]) - 1
    return np.array([x1 + w / 2, y1 + h / 2])


def get_size(info):
    w = eval(info[2]) - 1
    h = eval(info[3]) - 1
    return np.array([w, h])


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

        # calculate the change of 'size ratio' (w/h) from frame t-1 to frame t
        r_t = get_size(id_frame_info)[0] / get_size(id_frame_info)[1]
        r_t_1 = get_size(id_frame_info_t_1)[0] / get_size(id_frame_info_t_1)[1]
        dr = r_t / r_t_1

        # store the object size at frame t
        w, h = get_size(id_frame_info)

        # store the calculation results in the according frame in the dict 'all_info'
        if id_frame not in all_info.keys():
            all_info[id_frame] = []
        score = eval(id_frame_info[4].split(',')[0][-6:])
        all_info[id_frame].append([track_id, dv[0], dv[1], dl[0], dl[1], dr, score, w, h, int(id_frame_info[-1])])
        id_frame_info_t_2 = id_frame_info_t_1
        id_frame_info_t_1 = id_frame_info


# calculate the potential of a unary-type factor
def calc_unary(F_info, Wu):  # F_info is a list
    beta = 1
    score = F_info[6]
    dr = F_info[5]
    Pu_0 = abs(0. - score)
    Pu_1_s = abs(1. - score)
    Pu_1_r = abs(1 - dr)
    Pu_1 = Pu_1_s + beta * Pu_1_r
    return np.array([exp(-Wu * Pu_0), exp(-Wu * Pu_1)])
    # here we use parameter share, whereas we could also use different
    # parameters for each label configuration


def calc_binary(F_info, Wb):  # F_info is a list of two lists
    dvxi = F_info[0][1]
    dvyi = F_info[0][2]
    dlxi = F_info[0][3]
    dlyi = F_info[0][4]
    scorei = F_info[0][6]
    hi = F_info[0][8]

    dvxj = F_info[1][1]
    dvyj = F_info[1][2]
    dlxj = F_info[1][3]
    dlyj = F_info[1][4]
    scorej = F_info[1][6]
    hj = F_info[1][8]

    alpha1 = 2
    alpha2 = 10
    alpha3 = 10
    theta = 0.99
    tao = 1 / (hi + hj)
    Pb_11 = alpha1 * tao * ((dvxi - dvxj) ** 2 + (dvyi - dvyj) ** 2) \
            + alpha2 * (abs(dlxi - dlxj) + abs(dlyi - dlyj))
    if (scorei > theta and scorej > theta):
        Pb_not_11 = alpha3
    else:
        Pb_not_11 = 0
    return np.array([[exp(-Wb * Pb_not_11), exp(-Wb * Pb_not_11)], [exp(-Wb * Pb_not_11), exp(-Wb * Pb_11)]])


test_frame_id = 691

for test_frame_id in range(frame[0], frame[1]):
    frame_info = all_info[test_frame_id]

    g = fg.Graph()

    node_id_list = [l[0] for l in frame_info]

    for i in range(len(node_id_list)):
        node_i = str(node_id_list[i])  # node_i is the track_id (turned into str) for the variable node i
        g.rv(node_i, 2)  # yi can be 0 or 1
    print(f"\n node_id_list:{node_id_list}")
    for i in range(len(node_id_list)):
        node_i = str(node_id_list[i])
        if node_i == '23':
            test = 0
        # first, add the unary factors
        # frame_info[i] -> [id,dvx,dvy,dl,score,is_fake]
        g.factor([node_i], potential=calc_unary(frame_info[i], Wu=1))
        # second, add the binary factors
        for j in range(len(node_id_list)):
            if j <= i:
                continue
            node_j = str(node_id_list[j])

            g.factor([node_i, node_j], potential=calc_binary([frame_info[i], frame_info[j]], Wb=0.05))

    # Run (loopy) belief propagation (LBP)
    iters, converged = g.lbp_MAP(normalize=True)

    marg_tuples = g.rv_marginals(normalize=False)

    print(f"marg_tuples: {marg_tuples}")

    for rv, marg in marg_tuples:
        #   print(str(rv))
        vals = range(rv.n_opts)
        if len(rv.labels) > 0:
            vals = rv.labels
        map_rv = np.argmax(marg)
    #    print('\t', map_rv)
    # print('\n LBP ran for %d iterations. Converged = %r' % (iters, converged))
    # print('\n')
    print(f'frame id:{test_frame_id} in ' + dataset)
    # # Print out the final messages from LBP
    g.print_messages()
    # # print('\n')
    # # Print out the final marginals
    g.print_rv_marginals(normalize=True)
    g.print_rv_MAP()


class RcFilter(object):
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.inited = False
        self.state_ = 0

    def rc_filter(self, state):
        if self.inited:
            self.state_ = state + self.alpha * (self.state_ - state)
        else:
            self.state_ = state
            self.inited = True

        return self.state_

    def set_alpha(self, alpha):
        self.alpha = alpha

    def reset(self, alpha):
        self.inited = False
        self.alpha = alpha


if __name__ == "__main__":
    t4 = [688.3897, 374.0476, 868.1024, 785.6213]
    t3 = [661.3889, 375.3804, 857.5449, 794.2615]
    t2 = [657.6810, 375.7964, 847.5156, 801.2861]
    t1 = [659.8661, 377.5836, 843.6268, 795.8494]

    a = (t4[2] + t4[0]) * 0.5 + (t2[2] + t2[0]) * 0.5 - (t3[2] + t3[0])
    b = (t4[2] + t4[0]) * 0.5 - (t3[2] + t3[0]) * 0.5
    c = (t3[2] + t3[0]) * 0.5 - (t2[2] + t2[0]) * 0.5

    print(a)
    print(c)
    print(b)
    rc = RcFilter()

    c_ = rc.rc_filter(c)
    rc.set_alpha(0.2)
    b_ = rc.rc_filter(b)
    print(c_)
    print(b_)
