from collections import deque

from math import exp
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from scipy.optimize import linear_sum_assignment
import cv2

from .utils import bbox_overlaps, warp_pos, get_center, get_height, get_width, make_pos

from torchvision.ops.boxes import clip_boxes_to_image, nms

import tracktor.factorgraph as fg


class Tracker:
    """The main tracking file, here is where magic happens."""
    # only track pedestrian
    cl = 1

    def __init__(self, obj_detect, reid_network, tracker_cfg):
        self.obj_detect = obj_detect  # detector
        self.reid_network = reid_network
        self.detection_person_thresh = tracker_cfg['detection_person_thresh']
        self.regression_person_thresh = tracker_cfg['regression_person_thresh']
        self.detection_nms_thresh = tracker_cfg['detection_nms_thresh']
        self.regression_nms_thresh = tracker_cfg['regression_nms_thresh']
        self.public_detections = tracker_cfg['public_detections']
        self.inactive_patience = tracker_cfg['inactive_patience']
        self.do_reid = tracker_cfg['do_reid']
        self.max_features_num = tracker_cfg['max_features_num']
        self.reid_sim_threshold = tracker_cfg['reid_sim_threshold']
        self.reid_iou_threshold = tracker_cfg['reid_iou_threshold']
        self.do_align = tracker_cfg['do_align']
        self.motion_model_cfg = tracker_cfg['motion_model']

        self.warp_mode = eval(tracker_cfg['warp_mode'])
        self.number_of_iterations = tracker_cfg['number_of_iterations']
        self.termination_eps = tracker_cfg['termination_eps']

        self.tracks = []
        self.inactive_tracks = []
        self.track_num = 0
        self.im_index = 0
        self.results = {}

        # for crf_inference
        # self.tracks_count = 0
        self.id_frame_info_t_2 = []
        self.id_frame_info_t_1 = []
        self.id_frame_info_t = []
        self.all_metric_info = []

    def reset(self, hard=True):
        """reset for different seq"""
        self.tracks = []
        self.inactive_tracks = []

        # self.tracks_count = 0
        self.id_frame_info_t_2 = []
        self.id_frame_info_t_1 = []
        self.id_frame_info_t = []
        self.all_metric_info = []

        if hard:
            self.track_num = 0
            self.results = {}
            self.im_index = 0

    # functions for crf inference
    # def get_center(self, pos):
    #     """[x1,y1,x2,y2]->[cx,cy]"""
    #     cx = (pos[:, 0] + pos[:, 2]) / 2.
    #     cy = (pos[:, 1] + pos[:, 3]) / 2.
    #     center = torch.stack([cx, cy], axis=1)
    #     return center
    #
    # def get_size(self, boxes):
    #     """[x1,y1,x2,y2]->[w,h]."""
    #     w = boxes[:, 2] - boxes[:, 0]
    #     h = boxes[:, 3] - boxes[:, 1]
    #     box_size = torch.stack([w, h], axis=1)
    #     return box_size

    # -------

    def tracks_to_inactive(self, tracks):
        self.tracks = [t for t in self.tracks if t not in tracks]  # update tracks
        for t in tracks:
            t.pos = t.last_pos[-1]
        self.inactive_tracks += tracks

    def add(self, new_det_pos, new_det_scores, new_det_features):
        """Initializes new Track objects and saves them."""
        num_new = new_det_pos.size(0)
        for i in range(num_new):
            self.tracks.append(Track(
                new_det_pos[i].view(1, -1),
                new_det_scores[i],
                self.track_num + i,
                new_det_features[i].view(1, -1),
                self.inactive_patience,
                self.max_features_num,
                self.motion_model_cfg['n_steps'] if self.motion_model_cfg['n_steps'] > 0 else 1
            ))
        self.track_num += num_new

    def get_crf_metrics(self, track):
        """calculate crf metrics return list of [track_id,dv_x,dv_y,dl_x,dl_y,dr,score,w,h]"""

        last_pos = list(track.last_pos)
        track_id = int(track.id)

        # calculate the change of velocity from frame t-1 to frame t
        v_t = get_center(last_pos[-1]) - get_center(last_pos[-2])
        v_t_1 = get_center(last_pos[-2]) - get_center(last_pos[-3])
        dv = v_t - v_t_1

        # calculate the change of 'size change' from frame t-1 to frame t(temporary code,too complex)
        w_t = get_width(last_pos[-1])
        w_t_1 = get_width(last_pos[-2])
        w_t_2 = get_width(last_pos[-3])
        dw_t = w_t / w_t_1
        dw_t_1 = w_t_1 / w_t_2
        dw = dw_t / dw_t_1
        h_t = get_height(last_pos[-1])
        h_t_1 = get_height(last_pos[-2])
        h_t_2 = get_height(last_pos[-3])
        dh_t = h_t / h_t_1
        dh_t_1 = h_t_1 / h_t_2
        dh = dh_t / dh_t_1

        # calculate the change of `size ratio`(w/h) from frame t-1 to frame t
        r_t = w_t / h_t
        r_t_1 = w_t_1 / h_t_1
        dr = r_t / r_t_1

        crf_metric = [track_id, dv[0], dv[1], dw, dh, dr, track.score, w_t, h_t, track.track_count]
        return crf_metric

    # calculate the potential of a unary-type factor
    def calc_unary(self, F_info, Wu):
        beta = 1.
        score = F_info[6].detach().cpu().numpy()
        dr = F_info[5].detach().cpu().numpy()
        Pu_0 = abs(0. - score)
        Pu_1_s = abs(1. - score)
        Pu_1_r = abs(1. - dr)
        Pu_1 = Pu_1_s + beta * Pu_1_r
        return np.array([exp(-Wu * Pu_0), exp(-Wu * Pu_1)])
        # here we use parameter share, whereas we could also use different
        # parameters for each label configuration

    def calc_binary(self,F_info, Wb):  # F_info is a list of two lists
        dvxi = F_info[0][1].detach().cpu().numpy()
        dvyi = F_info[0][2].detach().cpu().numpy()
        dlxi = F_info[0][3].detach().cpu().numpy()
        dlyi = F_info[0][4].detach().cpu().numpy()
        scorei = F_info[0][6].detach().cpu().numpy()
        hi = F_info[0][8].detach().cpu().numpy()

        dvxj = F_info[1][1].detach().cpu().numpy()
        dvyj = F_info[1][2].detach().cpu().numpy()
        dlxj = F_info[1][3].detach().cpu().numpy()
        dlyj = F_info[1][4].detach().cpu().numpy()
        scorej = F_info[1][6].detach().cpu().numpy()
        hj = F_info[1][8].detach().cpu().numpy()

        alpha1 = 2.
        alpha2 = 10.
        alpha3 = 100.
        theta = 0.99
        tao = 1 / (hi + hj)
        Pb_11 = alpha1 * tao * ((dvxi - dvxj) ** 2 + (dvyi - dvyj) ** 2) \
                + alpha2 * (abs(dlxi - dlxj) + abs(dlyi - dlyj))
        if (scorei > theta and scorej > theta):
            Pb_not_11 = torch.FloatTensor([alpha3])
        else:
            Pb_not_11 = torch.FloatTensor([0.])
        return np.array([[exp(-Wb * Pb_not_11), exp(-Wb * Pb_not_11)], [exp(-Wb * Pb_not_11), exp(-Wb * Pb_11)]])

    def crf_inference(self, frame_info):
        '''
        :param frame_info:
        :return: tuple of marg for each node
        '''
        g = fg.Graph()
        node_id_list = [l[0] for l in frame_info]
        print(f"\n node_id_list:{node_id_list}")
        
        for i in range(len(node_id_list)):
            node_i = str(node_id_list[i])
            g.rv(node_i, 2)

        for i in range(len(node_id_list)):
            node_i = str(node_id_list[i])
            # first, add the unary factors
            # frame_info[i] -> [id,dvx,dvy,dw,dh,dr,score,w,h,count]
            g.factor([node_i], potential=self.calc_unary(frame_info[i], Wu=1))
            # second , add the binary factors
            for j in range(len(node_id_list)):
                if j >= i:
                    continue
                node_j = str(node_id_list[j])
                #print(f"node_i:{node_i},node_j:{node_j}\n")
                g.factor([node_i, node_j], potential=self.calc_binary([frame_info[i],frame_info[j]], Wb=0.05))

        # Run(loopy) belief propagation (LBP)
        iters, converged = g.lbp_MAP(normalize=True)
        marg_tuples = g.rv_marginals(normalize=False)

        return marg_tuples

    def regress_tracks(self, blob):
        """Regress the position of the tracks and also checks their scores."""
        pos = self.get_pos()
        pos_now = pos
        all_metric_info = []
        # regress
        boxes, scores = self.obj_detect.predict_boxes(blob['img'], pos)  # raw boxes -> (x1,y1,x2,y2) N*4
        pos = clip_boxes_to_image(boxes, blob['img'].shape[-2:])

        crf_s = []
        s = []
        # note: pos,scores is one-to-one correspondence
        for i in range(len(self.tracks) - 1, -1, -1):
        #for i in range(len(self.tracks)):
            t = self.tracks[i]
            t.score = scores[i]

            # get crf metrics
            if t.track_count > 2:
                crf_metric = self.get_crf_metrics(t)
                all_metric_info.append(crf_metric)  # append per track crf metric of this frame
            # ----

        # do crf inference
        if len(all_metric_info):
            marg_tuples = self.crf_inference(all_metric_info)
            for t_id, marg in marg_tuples:
                vals = range(t_id.n_opts)
                if len(t_id.labels) > 0:
                    vals = t_id.labels
                map_rv = np.argmax(marg)
                if map_rv == 0:
                    print(f"crf inference inactive  ----->  track_id:{t_id} \n") #print inactive track by crf inference with score
                    self.tracks_to_inactive([inactive_t for inactive_t in self.tracks if inactive_t.id == eval(t_id.name)])
                else:
                    crf_s.append(active_t.score for active_t in self.tracks if active_t.id == eval(t_id.name))
        else:
            print(" the crf metric lists is empty,skip crf inference and just use score instead")
            # use score metric for inactive track ,
        print(f"raw active tracks score: {scores} \n")
        print(f"crf active tracks score: {crf_s} \n")

        for i in range(len(self.tracks) - 1, -1, -1):
            t = self.tracks[i]
            t.score = scores[i] ##note :: scores need to update after inactive
            if scores[i] <= self.regression_person_thresh:
                self.tracks_to_inactive([t])
            else:
                s.append(scores[i])
                # t.prev_pos = t.pos
                t.pos = pos[i].view(1, -1)

        return torch.Tensor(s[::-1]).cuda()

    def get_pos(self):
        """Get the positions of all active tracks."""
        if len(self.tracks) == 1:
            pos = self.tracks[0].pos
        elif len(self.tracks) > 1:
            pos = torch.cat([t.pos for t in self.tracks], 0)
        else:
            pos = torch.zeros(0).cuda()
        return pos

    def get_features(self):
        """Get the features of all active tracks."""
        if len(self.tracks) == 1:
            features = self.tracks[0].features
        elif len(self.tracks) > 1:
            features = torch.cat([t.features for t in self.tracks], 0)
        else:
            features = torch.zeros(0).cuda()
        return features

    def get_inactive_features(self):
        """Get the features of all inactive tracks."""
        if len(self.inactive_tracks) == 1:
            features = self.inactive_tracks[0].features
        elif len(self.inactive_tracks) > 1:
            features = torch.cat([t.features for t in self.inactive_tracks], 0)
        else:
            features = torch.zeros(0).cuda()
        return features

    def reid(self, blob, new_det_pos, new_det_scores):
        """Tries to ReID inactive tracks with provided detections."""
        new_det_features = [torch.zeros(0).cuda() for _ in range(len(new_det_pos))]

        if self.do_reid:
            new_det_features = self.reid_network.test_rois(
                blob['img'], new_det_pos).data

            if len(self.inactive_tracks) >= 1:
                # calculate appearance distances
                dist_mat, pos = [], []
                for t in self.inactive_tracks:
                    dist_mat.append(torch.cat([t.test_features(feat.view(1, -1)) for feat in new_det_features], dim=1))
                    pos.append(t.pos)
                if len(dist_mat) > 1:
                    dist_mat = torch.cat(dist_mat, 0)
                    pos = torch.cat(pos, 0)
                else:
                    dist_mat = dist_mat[0]
                    pos = pos[0]

                # calculate IoU distances
                iou = bbox_overlaps(pos, new_det_pos)
                iou_mask = torch.ge(iou, self.reid_iou_threshold)
                iou_neg_mask = ~iou_mask
                # make all impossible assignments to the same add big value
                dist_mat = dist_mat * iou_mask.float() + iou_neg_mask.float() * 1000
                dist_mat = dist_mat.cpu().numpy()

                row_ind, col_ind = linear_sum_assignment(dist_mat)

                assigned = []
                remove_inactive = []
                for r, c in zip(row_ind, col_ind):
                    if dist_mat[r, c] <= self.reid_sim_threshold:
                        t = self.inactive_tracks[r]
                        self.tracks.append(t)
                        t.count_inactive = 0
                        t.pos = new_det_pos[c].view(1, -1)
                        t.reset_last_pos()
                        t.add_features(new_det_features[c].view(1, -1))
                        assigned.append(c)
                        remove_inactive.append(t)

                for t in remove_inactive:
                    self.inactive_tracks.remove(t)

                keep = torch.Tensor([i for i in range(new_det_pos.size(0)) if i not in assigned]).long().cuda()
                if keep.nelement() > 0:
                    new_det_pos = new_det_pos[keep]
                    new_det_scores = new_det_scores[keep]
                    new_det_features = new_det_features[keep]
                else:
                    new_det_pos = torch.zeros(0).cuda()
                    new_det_scores = torch.zeros(0).cuda()
                    new_det_features = torch.zeros(0).cuda()

        return new_det_pos, new_det_scores, new_det_features

    def get_appearances(self, blob):
        """Uses the siamese CNN to get the features for all active tracks."""
        new_features = self.reid_network.test_rois(blob['img'], self.get_pos()).data
        return new_features

    def add_features(self, new_features):
        """Adds new appearance features to active tracks."""
        for t, f in zip(self.tracks, new_features):
            t.add_features(f.view(1, -1))

    def align(self, blob):
        """Aligns the positions of active and inactive tracks depending on camera motion."""
        if self.im_index > 0:
            im1 = np.transpose(self.last_image.cpu().numpy(), (1, 2, 0))
            im2 = np.transpose(blob['img'][0].cpu().numpy(), (1, 2, 0))
            im1_gray = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)
            im2_gray = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY)
            warp_matrix = np.eye(2, 3, dtype=np.float32)
            criteria = (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, self.number_of_iterations, self.termination_eps)
            cc, warp_matrix = cv2.findTransformECC(im1_gray, im2_gray, warp_matrix, self.warp_mode, criteria)
            warp_matrix = torch.from_numpy(warp_matrix)

            for t in self.tracks:
                t.pos = warp_pos(t.pos, warp_matrix)
            # t.pos = clip_boxes(Variable(pos), blob['im_info'][0][:2]).data

            if self.do_reid:
                for t in self.inactive_tracks:
                    t.pos = warp_pos(t.pos, warp_matrix)

            if self.motion_model_cfg['enabled']:
                for t in self.tracks:
                    for i in range(len(t.last_pos)):
                        t.last_pos[i] = warp_pos(t.last_pos[i], warp_matrix)

    def motion_step(self, track):
        """Updates the given track's position by one step based on track.last_v"""
        if self.motion_model_cfg['center_only']:
            center_new = get_center(track.pos) + track.last_v
            track.pos = make_pos(*center_new, get_width(track.pos), get_height(track.pos))
        else:
            track.pos = track.pos + track.last_v

    def motion(self):
        """Applies a simple linear motion model that considers the last n_steps steps."""
        for t in self.tracks:
            last_pos = list(t.last_pos)

            # avg velocity between each pair of consecutive positions in t.last_pos
            if self.motion_model_cfg['center_only']:
                vs = [get_center(p2) - get_center(p1) for p1, p2 in zip(last_pos, last_pos[1:])]
            else:
                vs = [p2 - p1 for p1, p2 in zip(last_pos, last_pos[1:])]

            t.last_v = torch.stack(vs).mean(dim=0)
            self.motion_step(t)

        if self.do_reid:
            for t in self.inactive_tracks:
                if t.last_v.nelement() > 0:
                    self.motion_step(t)

    def step(self, blob):
        """This function should be called every timestep to perform tracking with a blob
        containing the image information.
        """
        for t in self.tracks:
            # add current position to last_pos list
            t.last_pos.append(t.pos.clone())
            t.track_count += 1

        ###########################
        # Look for new detections #
        ###########################

        # self.obj_detect.load_image(blob['data'][0])
        if self.public_detections:
            dets = blob['dets'].squeeze(dim=0)
            if dets.nelement() > 0:
                boxes, scores = self.obj_detect.predict_boxes(blob['img'], dets)
            else:
                boxes = scores = torch.zeros(0).cuda()
        else:
            boxes, scores = self.obj_detect.detect(blob['img'])

        if boxes.nelement() > 0:
            boxes = clip_boxes_to_image(boxes, blob['img'].shape[-2:])

            # Filter out tracks that have too low person score
            inds = torch.gt(scores, self.detection_person_thresh).nonzero().view(-1)
        else:
            inds = torch.zeros(0).cuda()

        if inds.nelement() > 0:
            det_pos = boxes[inds]

            det_scores = scores[inds]
        else:
            det_pos = torch.zeros(0).cuda()
            det_scores = torch.zeros(0).cuda()

        ##################
        # Predict tracks #
        ##################

        num_tracks = 0
        nms_inp_reg = torch.zeros(0).cuda()
        if len(self.tracks):
            # align
            if self.do_align:
                self.align(blob)

            # apply motion model
            if self.motion_model_cfg['enabled']:
                self.motion()
                self.tracks = [t for t in self.tracks if t.has_positive_area()]

            # regress
            person_scores = self.regress_tracks(blob)

            if len(self.tracks):
                # create nms input

                # nms here if tracks overlap
                keep = nms(self.get_pos(), person_scores, self.regression_nms_thresh)
                print(f"tracks:{len(self.tracks)}\n")
                print(f"keep tracks index:{keep}\n")
                self.tracks_to_inactive([self.tracks[i] for i in list(range(len(self.tracks))) if i not in keep])

                if keep.nelement() > 0:
                    if self.do_reid:
                        new_features = self.get_appearances(blob)
                        self.add_features(new_features)

        #####################
        # Create new tracks #
        #####################

        # !!! Here NMS is used to filter out detections that are already covered by tracks. This is
        # !!! done by iterating through the active tracks one by one, assigning them a bigger score
        # !!! than 1 (maximum score for detections) and then filtering the detections with NMS.
        # !!! In the paper this is done by calculating the overlap with existing tracks, but the
        # !!! result stays the same.
        if det_pos.nelement() > 0:
            keep = nms(det_pos, det_scores, self.detection_nms_thresh)
            det_pos = det_pos[keep]
            det_scores = det_scores[keep]

            # check with every track in a single run (problem if tracks delete each other)
            for t in self.tracks:
                nms_track_pos = torch.cat([t.pos, det_pos])
                nms_track_scores = torch.cat(
                    [torch.tensor([2.0]).to(det_scores.device), det_scores])
                keep = nms(nms_track_pos, nms_track_scores, self.detection_nms_thresh)

                keep = keep[torch.ge(keep, 1)] - 1

                det_pos = det_pos[keep]
                det_scores = det_scores[keep]
                if keep.nelement() == 0:
                    break

        if det_pos.nelement() > 0:
            new_det_pos = det_pos
            new_det_scores = det_scores

            # try to reidentify tracks
            new_det_pos, new_det_scores, new_det_features = self.reid(blob, new_det_pos, new_det_scores)

            # add new
            if new_det_pos.nelement() > 0:
                self.add(new_det_pos, new_det_scores, new_det_features)

        ####################
        # Generate Results #
        ####################

        for t in self.tracks:
            if t.id not in self.results.keys():
                self.results[t.id] = {}
            self.results[t.id][self.im_index] = np.concatenate([t.pos[0].cpu().numpy(), np.array([t.score])])

        for t in self.inactive_tracks:
            t.count_inactive += 1

        self.inactive_tracks = [
            t for t in self.inactive_tracks if t.has_positive_area() and t.count_inactive <= self.inactive_patience
        ]

        # for t in self.tracks:
        #     t.track_count += 1

        self.im_index += 1
        self.last_image = blob['img'][0]

    def get_results(self):
        return self.results


class Track(object):
    """This class contains all necessary for every individual track."""

    def __init__(self, pos, score, track_id, features, inactive_patience, max_features_num, mm_steps):
        self.id = track_id
        self.pos = pos
        self.score = score
        self.features = deque([features])
        self.ims = deque([])
        self.count_inactive = 0
        self.inactive_patience = inactive_patience
        self.max_features_num = max_features_num
        self.last_pos = deque([pos.clone()], maxlen=mm_steps + 2)
        self.last_v = torch.Tensor([])
        self.gt_id = None

        self.track_count = 0  # success track count >=3 ->crf_inference
        self.llast_post = deque([pos.clone()], maxlen=mm_steps + 2)

    def has_positive_area(self):
        return self.pos[0, 2] > self.pos[0, 0] and self.pos[0, 3] > self.pos[0, 1]

    def add_features(self, features):
        """Adds new appearance features to the object."""
        self.features.append(features)
        if len(self.features) > self.max_features_num:
            self.features.popleft()

    def test_features(self, test_features):
        """Compares test_features to features of this Track object"""
        if len(self.features) > 1:
            features = torch.cat(list(self.features), dim=0)
        else:
            features = self.features[0]
        features = features.mean(0, keepdim=True)
        dist = F.pairwise_distance(features, test_features, keepdim=True)
        return dist

    def reset_last_pos(self):
        self.last_pos.clear()
        self.track_count = 0
        self.last_pos.append(self.pos.clone())

    def reset_llast_pos(self):
        self.llast_post.clear()
        self.llast_post.append(self.pos.clone())
