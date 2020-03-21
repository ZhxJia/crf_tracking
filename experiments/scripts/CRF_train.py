from math import exp
import numpy as np
import factorgraph as fg

root = '/content/drive/My Drive/workspace/tracking_wo_bnw/output/tracktor/MOT17/Tracktor++/'


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
with open(root + 'MOT17-05-FRCNN.txt', 'r') as f:
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


#calculate the potential of a unary-type factor
def calc_unary(F_info,Wu): # F_info is a list
    score=F_info[4]
    Pu_0=abs(0-max(2*score-1,0))
    Pu_1 = abs(1. - max(2. * score - 1., 0))
    return np.array([exp(-Wu * Pu_0), exp(-Wu * Pu_1)])
    # here we use parameter share, whereas we could also use different
    # parameters for each label configuration

def calc_unary_phi(F_info): # F_info is a list
    score=F_info[4]
    Pu_0=abs(0-max(2*score-1,0))
    Pu_1 = abs(1. - max(2. * score - 1., 0))
    return np.array([Pu_0, Pu_1])

def calc_binary(F_info,Wb): # F_info is a list of two lists
    dvxi=F_info[0][1]
    dvxj=F_info[1][1]
    dvyi=F_info[0][2]
    dvyj=F_info[1][2]
    dli=F_info[0][3]
    dlj=F_info[1][3]
    alpha1 = 0.025
    alpha2=10
    Pb_00_11 = alpha1*((dvxi - dvxj) ** 2+(dvyi - dvyj) ** 2) + alpha2*abs(dli - dlj)
    return np.array([[exp(-Wb*Pb_00_11),exp(-0)],[exp(-0),exp(-Wb*Pb_00_11)]])

def calc_binary_phi(F_info): # F_info is a list of two lists
    dvxi=F_info[0][1]
    dvxj=F_info[1][1]
    dvyi=F_info[0][2]
    dvyj=F_info[1][2]
    dli=F_info[0][3]
    dlj=F_info[1][3]
    alpha1 = 0.025
    alpha2=10
    Pb_00_11 = alpha1*((dvxi - dvxj) ** 2+(dvyi - dvyj) ** 2) + alpha2*abs(dli - dlj)
    return np.array([[Pb_00_11,0],[0,Pb_00_11]])

def calc_Fu_marginal(F,y, normalize=True): # y is the actual value of F
    '''
    Gets the marginal for the factor when taking F_value=y
    '''

    potential=F.get_potential()[y]
    total_msg = 1
    for i, rv in enumerate(F._rvs): # F._rvs is a list
        total_msg *= rv.get_outgoing_for(F)[y]
    marg=potential*total_msg # actually, it is 'exp(marg_bar)' in (3.12)

    if normalize:
        zF=0
        for y in [0,1]:
            potential = F.get_potential()[y]
            total_msg = 1
            for i, rv in enumerate(F._rvs):  # F._rvs is a list
                total_msg *= rv.get_outgoing_for(F)[y]
            zF+=potential*total_msg
        marg /= zF

    return marg

def calc_Fb_marginal(F,y, normalize=True):
    '''
    Gets the marginal for the factor when F_value=y
    '''

    potential=F.get_potential()[y//2,y%2]
    total_msg = 1
    y_list=[y//2,y%2]
    for i, rv in enumerate(F._rvs): # F._rvs is a list
        total_msg *= rv.get_outgoing_for(F)[y_list[i]]
    marg=potential*total_msg # actually, it is 'exp(marg_bar)' in (3.12)

    if normalize:
        zF=0
        for y in [0,1,2,3]:
            potential = F.get_potential()[y // 2, y % 2]
            total_msg = 1
            y_list = [y // 2, y % 2]
            for i, rv in enumerate(F._rvs):  # F._rvs is a list
                total_msg *= rv.get_outgoing_for(F)[y_list[i]]
            zF+=potential*total_msg
        marg /= zF

    return marg

test_frame_id=567

frame_info=all_info[test_frame_id]

# W -> [unary, binary]
W=[4,0.2]
batch=[frame_info]
num_epochs=1
lr=1e-2
end_lr=1e-3
decay_rate=(lr-end_lr)/num_epochs
for epoch in range(num_epochs):
    lr-=num_epochs*decay_rate
    for sample in batch: # 'sample' plays a similar role to 'frame_info'
        g = Graph()
        node_id_list = [l[0] for l in sample]
        for i in range(len(node_id_list)):
            node_i = str(node_id_list[i])  # node_i is the track_id (turned into str) for the variable node i
            g.rv(node_i, 2)  # yi can be 0 or 1

        for i in range(len(node_id_list)):
            node_i = str(node_id_list[i])
            # first, add the unary factors
            # frame_info[i] -> [id,dvx,dvy,dl,score,is_fake]
            g.factor([node_i], potential=calc_unary(sample[i], W[0]))
            # second, add the binary factors
            for j in range(len(node_id_list)):
                if j <= i:
                    continue
                node_j = str(node_id_list[j])
                g.factor([node_i, node_j], potential=calc_binary([sample[i], sample[j]], W[1]))
        iters,converged=g.lbp(normalize=True)
        phi_u=0
        E_phi_u=0
        phi_b=0
        E_phi_b=0
        unary_factors=[]
        binary_factors=[]
        # calc phi and split factors
        for factor in g.get_factors():
            if len(factor._rvs)==1: # factors with one rv, i.e. unary factory
                rv_ind=node_id_list.index(eval(factor._rvs[0].name))
                # the information corresponding to the index of rv in node_id_list
                sample_rv=sample[rv_ind]
                phi_u+=calc_unary_phi(sample_rv)[sample_rv[-1]] #sample_rv[-1] is the label of rv
                unary_factors.append(factor)
            else: # binary factors
                rv_inds=[]
                for i in range(2):
                    rv_inds.append(node_id_list.index(eval(factor._rvs[i].name)))
                sample_rvs=[sample[rv_inds[0]],sample[rv_inds[1]]]
                phi_b+=calc_binary_phi(sample_rvs)[sample_rvs[0][-1],sample_rvs[1][-1]]
                binary_factors.append(factor)
        # calc E_phi
        for factor in unary_factors:
            rv_ind = node_id_list.index(eval(factor._rvs[0].name))
            phi_u_y = 0  # sum of phi_u with a fixed y
            sum_over_yc=0
            for y in [0,1]:
                sample_rv = sample[rv_ind]
                phi_u_y = calc_unary_phi(sample_rv)[y]
                Fu_marginal_y=calc_Fu_marginal(factor,y)
                sum_over_yc+=phi_u_y*Fu_marginal_y
            E_phi_u+=sum_over_yc

        for factor in binary_factors:
            rv_inds = []
            for i in range(2):
                rv_inds.append(node_id_list.index(eval(factor._rvs[i].name)))
            phi_b_y=0
            sum_over_yc=0
            for y in [0,1,2,3]:
                sample_rvs=[sample[rv_inds[0]],sample[rv_inds[1]]]
                phi_b_y=calc_binary_phi(sample_rvs)[y//2,y%2]
                Fb_marginal_y=calc_Fb_marginal(factor,y)
                sum_over_yc+=phi_b_y*Fb_marginal_y
            E_phi_b+=sum_over_yc

        # update the weights
        dWu=phi_u-E_phi_u
        dWb=phi_b-E_phi_b
        W[0]=W[0]-lr*dWu
        W[1]=W[1]-lr*dWb