import csv
import numpy as np

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
    return np.array([w, h])


# [frame+1, i+1, x1+1, y1+1, x2-x1+1, y2-y1+1, str(torch.tensor(score)), is_fake]
with open(root + 'MOT17-02-FRCNN.txt', 'r') as f:
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

        # store the calculation results in the according frame in the dict 'all_info'
        if id_frame not in all_info.keys():
            all_info[id_frame] = []
        score = eval(id_frame_info[4].split(',')[0][-6:])
        all_info[id_frame].append([track_id, dv[0], dv[1], dl[0],dl[1],dr, score, int(id_frame_info[-1])])
        id_frame_info_t_2 = id_frame_info_t_1
        id_frame_info_t_1 = id_frame_info

max_frame = max(frame for frame in all_info.keys())

with open(root + 'seq_02.txt', 'w') as f:
    writer = csv.writer(f, delimiter=',')
    for frame in range(3, max_frame + 1):
        print('\n Frame:{}'.format(frame), file=f)
        if frame in all_info.keys():
            frame_info = all_info[frame]
            for line in frame_info:
                writer.writerow([line])  # id,dvx,dvy,dlx,dly,w/h_ratio,score,is_fake
        else:
            writer.writerow([])  # some frames may contain nothing
