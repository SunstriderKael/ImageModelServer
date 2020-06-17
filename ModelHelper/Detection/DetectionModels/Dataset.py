from torch.utils import data
from ModelHelper.Common.CommonUtils import get, get_valid
from ModelHelper.Common.CommonUtils.HandleImage import get_img_list
from shapely.geometry import Polygon
from ModelHelper.Detection.Component.geo_map_cython_lib import gen_geo_map
import os
import random
import numpy as np
import cv2
import math
import pyclipper


class EastDataset(data.Dataset):
    def __init__(self, **kwargs):
        self.folder = get_valid('folder', kwargs)
        self.transforms = get('transforms', kwargs, None)
        self.input_size = get('input_size', kwargs, (768, 768))
        self.detection_transforms = get('detection_transforms', kwargs, None)
        self.data_list = self.__get_data_list()

    def __get_data_list(self):
        assert os.path.exists(self.folder)
        img_list = get_img_list(self.folder)
        data_list = list()
        for img in img_list:
            data = dict()
            img_path = os.path.join(self.folder, img)
            img_name = os.path.splitext(img)[0]
            txt_file = img_name + '.txt'
            txt_path = os.path.join(self.folder, txt_file)
            if not os.path.exists(txt_path):
                raise RuntimeError('{} not exists!'.format(txt_path))
            else:
                data['img_path'] = img_path
                data['txt_path'] = txt_path
                data_list.append(data)
        return data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        status = True
        while status:
            try:
                data = self.data_list[idx]
                img, score_map, geo_map, training_mask, img_path = self.__get_label(data)
                status = False
            except BaseException as e:
                print('Error on get label in ModelHelper.Detection.DetectionModels.Dataset: {}'.format(e))
            if status is True:
                idx = random.randint(0, len(self.data_list))
        img = img.astype(np.float32)
        img = self.transforms(img)
        output_data = dict()
        output_data['img'] = img
        output_data['score_map'] = score_map
        output_data['geo_map'] = geo_map
        output_data['training_mask'] = training_mask
        output_data['img_path'] = img_path
        return output_data

    def __get_label(self, data):
        img_path = data['img_path']
        txt_path = data['txt_path']

        img = cv2.imread(img_path)
        h, w, _ = img.shape
        text_polys, text_tags = self.__load_annoataion(txt_path)
        text_polys, text_tags = self.__check_and_validate_polys(text_polys, text_tags, (h, w))

        if self.detection_transforms is not None:
            img, text_polys = self.detection_transforms(img, text_polys)

        new_h, new_w, _ = img.shape
        resize_h = self.input_size[0]
        resize_w = self.input_size[1]
        # resize the image to input size
        img = cv2.resize(img, dsize=(resize_w, resize_h))

        if text_polys.shape[0] == 0:
            score_map = np.zeros((resize_h, resize_w))
            geo_map = np.zeros((resize_h, resize_w, 5))
            training_mask = np.zeros((resize_h, resize_w))
            score_map = score_map[np.newaxis, ::4, ::4].astype(np.float32)
            geo_map = geo_map[::4, ::4, :].astype(np.float32)
            training_mask = training_mask[np.newaxis, ::4, ::4].astype(np.float32)

        else:
            resize_ratio_3_x = resize_w / float(new_w)
            resize_ratio_3_y = resize_h / float(new_h)

            text_polys[:, :, 0] *= resize_ratio_3_x
            text_polys[:, :, 1] *= resize_ratio_3_y
            score_map, geo_map, training_mask = self.__generate_rbox((resize_h, resize_w), text_polys, text_tags)
            score_map = score_map[np.newaxis, ::4, ::4].astype(np.float32)
            geo_map = geo_map[::4, ::4, :].astype(np.float32)
            training_mask = training_mask[np.newaxis, ::4, ::4].astype(np.float32)
        return img, score_map, geo_map, training_mask, img_path

    @staticmethod
    def __load_annoataion(txt_path):
        '''
        load annotation from the text file

        Note:
        modified
        1. top left vertice
        2. clockwise

        :param p:
        :return:
        '''
        text_polys = []
        text_tags = []
        if not os.path.exists(txt_path):
            return np.array(text_polys, dtype=np.float32)
        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.encode('utf-8').decode('utf-8-sig')
                splited = line.strip().split(',')
                x1, y1 = splited[0], splited[1]
                x2, y2 = splited[2], splited[3]
                x3, y3 = splited[4], splited[5]
                x4, y4 = splited[6], splited[7]

                text_polys.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
                label = 'label'
                if label == '*' or label == '###':
                    text_tags.append(True)
                else:
                    text_tags.append(False)

        return np.array(text_polys, dtype=np.float32), np.array(text_tags, dtype=np.bool)

    def __check_and_validate_polys(self, polys, tags, xxx_todo_changeme):
        '''
        check so that the text poly is in the same direction,
        and also filter some invalid polygons
        :param polys:
        :param tags:
        :return:
        '''

        (h, w) = xxx_todo_changeme
        if polys.shape[0] == 0:
            return polys
        polys[:, :, 0] = np.clip(polys[:, :, 0], 0, w - 1)
        polys[:, :, 1] = np.clip(polys[:, :, 1], 0, h - 1)

        validated_polys = []
        validated_tags = []

        # find top-left and clockwise
        polys = self.__choose_best_begin_point(polys)

        for poly, tag in zip(polys, tags):
            p_area = self.__polygon_area(poly)
            if abs(p_area) < 1:
                # print poly
                # print('invalid poly')
                continue
            if p_area > 0:
                # print('poly in wrong direction')
                poly = poly[(0, 3, 2, 1), :]
            validated_polys.append(poly)
            validated_tags.append(tag)
        return np.array(validated_polys), np.array(validated_tags)

    def __choose_best_begin_point(self, pre_result):
        """
        find top-left vertice and resort
        """
        final_result = []
        for coordinate in pre_result:
            x1 = coordinate[0][0]
            y1 = coordinate[0][1]
            x2 = coordinate[1][0]
            y2 = coordinate[1][1]
            x3 = coordinate[2][0]
            y3 = coordinate[2][1]
            x4 = coordinate[3][0]
            y4 = coordinate[3][1]
            xmin = min(x1, x2, x3, x4)
            ymin = min(y1, y2, y3, y4)
            xmax = max(x1, x2, x3, x4)
            ymax = max(y1, y2, y3, y4)
            combinate = [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
                         [[x2, y2], [x3, y3], [x4, y4], [x1, y1]],
                         [[x3, y3], [x4, y4], [x1, y1], [x2, y2]],
                         [[x4, y4], [x1, y1], [x2, y2], [x3, y3]]]
            dst_coordinate = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
            force = 100000000.0
            force_flag = 0
            for i in range(4):
                temp_force = self.__calculate_distance(combinate[i][0], dst_coordinate[0]) + self.__calculate_distance(
                    combinate[i][1], dst_coordinate[1]) + self.__calculate_distance(combinate[i][2], dst_coordinate[
                    2]) + self.__calculate_distance(combinate[i][3], dst_coordinate[3])
                if temp_force < force:
                    force = temp_force
                    force_flag = i
            # if force_flag != 0:
            #    print("choose one direction!")
            final_result.append(combinate[force_flag])

        return final_result

    def __polygon_area(slef, poly):
        '''
        compute area of a polygon
        :param poly:
        :return:
        '''
        poly_ = np.array(poly)
        assert poly_.shape == (4, 2), 'poly shape should be 4,2'
        edge = [
            (poly[1][0] - poly[0][0]) * (poly[1][1] + poly[0][1]),
            (poly[2][0] - poly[1][0]) * (poly[2][1] + poly[1][1]),
            (poly[3][0] - poly[2][0]) * (poly[3][1] + poly[2][1]),
            (poly[0][0] - poly[3][0]) * (poly[0][1] + poly[3][1])
        ]
        return np.sum(edge) / 2.

    @staticmethod
    def __calculate_distance(c1, c2):
        return math.sqrt(math.pow(c1[0] - c2[0], 2) + math.pow(c1[1] - c2[1], 2))

    def __generate_rbox(self, im_size, polys, tags):
        """
        score map is (128, 128, 1) with shrinked poly
        poly mask is (128, 128, 1) with differnt colors


        geo map is  (128, 128, 5) with
        """
        h, w = im_size
        poly_mask = np.zeros((h, w), dtype=np.uint8)
        score_map = np.zeros((h, w), dtype=np.uint8)
        geo_map = np.zeros((h, w, 5), dtype=np.float32)
        # mask used during traning, to ignore some hard areas
        training_mask = np.ones((h, w), dtype=np.uint8)
        for poly_idx, poly_tag in enumerate(zip(polys, tags)):
            poly = poly_tag[0]
            tag = poly_tag[1]
            poly = np.array(poly)
            tag = np.array(tag)
            r = [None, None, None, None]
            for i in range(4):
                r[i] = min(np.linalg.norm(poly[i] - poly[(i + 1) % 4]),
                           np.linalg.norm(poly[i] - poly[(i - 1) % 4]))
            # score map
            shrinked_poly = self.__shrink_poly(poly.copy(), r).astype(np.int32)[np.newaxis, :, :]
            cv2.fillPoly(score_map, shrinked_poly, 1)

            # use different color to draw poly mask
            cv2.fillPoly(poly_mask, shrinked_poly, poly_idx + 1)
            # if the poly is too small, then ignore it during training
            poly_h = min(np.linalg.norm(poly[0] - poly[3]), np.linalg.norm(poly[1] - poly[2]))
            poly_w = min(np.linalg.norm(poly[0] - poly[1]), np.linalg.norm(poly[2] - poly[3]))
            # if min(poly_h, poly_w) < FLAGS.min_text_size:
            if min(poly_h, poly_w) < 10:
                cv2.fillPoly(training_mask, poly.astype(np.int32)[np.newaxis, :, :], 0)
            if tag:
                cv2.fillPoly(training_mask, poly.astype(np.int32)[np.newaxis, :, :], 0)

            xy_in_poly = np.argwhere(poly_mask == (poly_idx + 1))
            # if geometry == 'RBOX':
            # 对任意两个顶点的组合生成一个平行四边形
            fitted_parallelograms = []
            for i in range(4):
                p0 = poly[i]
                p1 = poly[(i + 1) % 4]
                p2 = poly[(i + 2) % 4]
                p3 = poly[(i + 3) % 4]

                # fit_line ([x1, x2], [y1, y2]) return k, -1, b just a line
                edge = self.__fit_line([p0[0], p1[0]], [p0[1], p1[1]])  # p0, p1
                backward_edge = self.__fit_line([p0[0], p3[0]], [p0[1], p3[1]])  # p0, p3
                forward_edge = self.__fit_line([p1[0], p2[0]], [p1[1], p2[1]])  # p1, p2

                # select shorter line
                if self.__point_dist_to_line(p0, p1, p2) > self.__point_dist_to_line(p0, p1, p3):
                    # 平行线经过p2
                    if edge[1] == 0:  # verticle
                        edge_opposite = [1, 0, -p2[0]]
                    else:
                        edge_opposite = [edge[0], -1, p2[1] - edge[0] * p2[0]]
                else:
                    # 经过p3
                    if edge[1] == 0:
                        edge_opposite = [1, 0, -p3[0]]
                    else:
                        edge_opposite = [edge[0], -1, p3[1] - edge[0] * p3[0]]
                # move forward edge
                new_p0 = p0
                new_p1 = p1
                new_p2 = p2
                new_p3 = p3
                new_p2 = self.__line_cross_point(forward_edge, edge_opposite)
                if self.__point_dist_to_line(p1, new_p2, p0) > self.__point_dist_to_line(p1, new_p2, p3):
                    # across p0
                    if forward_edge[1] == 0:
                        forward_opposite = [1, 0, -p0[0]]
                    else:
                        forward_opposite = [forward_edge[0], -1, p0[1] - forward_edge[0] * p0[0]]
                else:
                    # across p3
                    if forward_edge[1] == 0:
                        forward_opposite = [1, 0, -p3[0]]
                    else:
                        forward_opposite = [forward_edge[0], -1, p3[1] - forward_edge[0] * p3[0]]
                new_p0 = self.__line_cross_point(forward_opposite, edge)
                new_p3 = self.__line_cross_point(forward_opposite, edge_opposite)
                fitted_parallelograms.append([new_p0, new_p1, new_p2, new_p3, new_p0])
                # or move backward edge
                new_p0 = p0
                new_p1 = p1
                new_p2 = p2
                new_p3 = p3
                new_p3 = self.__line_cross_point(backward_edge, edge_opposite)
                if self.__point_dist_to_line(p0, p3, p1) > self.__point_dist_to_line(p0, p3, p2):
                    # across p1
                    if backward_edge[1] == 0:
                        backward_opposite = [1, 0, -p1[0]]
                    else:
                        backward_opposite = [backward_edge[0], -1, p1[1] - backward_edge[0] * p1[0]]
                else:
                    # across p2
                    if backward_edge[1] == 0:
                        backward_opposite = [1, 0, -p2[0]]
                    else:
                        backward_opposite = [backward_edge[0], -1, p2[1] - backward_edge[0] * p2[0]]
                new_p1 = self.__line_cross_point(backward_opposite, edge)
                new_p2 = self.__line_cross_point(backward_opposite, edge_opposite)
                fitted_parallelograms.append([new_p0, new_p1, new_p2, new_p3, new_p0])

            areas = [Polygon(t).area for t in fitted_parallelograms]
            parallelogram = np.array(fitted_parallelograms[np.argmin(areas)][:-1], dtype=np.float32)
            # sort thie polygon
            parallelogram_coord_sum = np.sum(parallelogram, axis=1)
            min_coord_idx = np.argmin(parallelogram_coord_sum)
            parallelogram = parallelogram[
                [min_coord_idx, (min_coord_idx + 1) % 4, (min_coord_idx + 2) % 4, (min_coord_idx + 3) % 4]]

            rectange = self.__rectangle_from_parallelogram(parallelogram)
            rectange, rotate_angle = self.__sort_rectangle(rectange)
            # print('parallel {} rectangle {}'.format(parallelogram, rectange))
            p0_rect, p1_rect, p2_rect, p3_rect = rectange
            # this is one area of many
            """
            for y, x in xy_in_poly:
                point = np.array([x, y], dtype=np.float32)
                # top
                geo_map[y, x, 0] = point_dist_to_line(p0_rect, p1_rect, point)
                # right
                geo_map[y, x, 1] = point_dist_to_line(p1_rect, p2_rect, point)
                # down
                geo_map[y, x, 2] = point_dist_to_line(p2_rect, p3_rect, point)
                # left
                geo_map[y, x, 3] = point_dist_to_line(p3_rect, p0_rect, point)
                # angle
                geo_map[y, x, 4] = rotate_angle
            """
            gen_geo_map.gen_geo_map(geo_map, xy_in_poly, rectange, rotate_angle)

        ###sum up
        # score_map , in shrinked poly is 1
        # geo_map, corresponding to score map
        # training map is less than geo_map

        return score_map, geo_map, training_mask

    @staticmethod
    def __shrink_poly(poly, r):
        '''
        fit a poly inside the origin poly, maybe bugs here...
        used for generate the score map
        :param poly: the text poly
        :param r: r in the paper
        :return: the shrinked poly
        '''
        # shrink ratio
        R = 0.1
        # find the longer pair
        if np.linalg.norm(poly[0] - poly[1]) + np.linalg.norm(poly[2] - poly[3]) > \
                np.linalg.norm(poly[0] - poly[3]) + np.linalg.norm(poly[1] - poly[2]):
            # first move (p0, p1), (p2, p3), then (p0, p3), (p1, p2)
            ## p0, p1
            theta = np.arctan2((poly[1][1] - poly[0][1]), (poly[1][0] - poly[0][0]))
            poly[0][0] += R * r[0] * np.cos(theta)
            poly[0][1] += R * r[0] * np.sin(theta)
            poly[1][0] -= R * r[1] * np.cos(theta)
            poly[1][1] -= R * r[1] * np.sin(theta)
            ## p2, p3
            theta = np.arctan2((poly[2][1] - poly[3][1]), (poly[2][0] - poly[3][0]))
            poly[3][0] += R * r[3] * np.cos(theta)
            poly[3][1] += R * r[3] * np.sin(theta)
            poly[2][0] -= R * r[2] * np.cos(theta)
            poly[2][1] -= R * r[2] * np.sin(theta)
            ## p0, p3
            theta = np.arctan2((poly[3][0] - poly[0][0]), (poly[3][1] - poly[0][1]))
            poly[0][0] += R * r[0] * np.sin(theta)
            poly[0][1] += R * r[0] * np.cos(theta)
            poly[3][0] -= R * r[3] * np.sin(theta)
            poly[3][1] -= R * r[3] * np.cos(theta)
            ## p1, p2
            theta = np.arctan2((poly[2][0] - poly[1][0]), (poly[2][1] - poly[1][1]))
            poly[1][0] += R * r[1] * np.sin(theta)
            poly[1][1] += R * r[1] * np.cos(theta)
            poly[2][0] -= R * r[2] * np.sin(theta)
            poly[2][1] -= R * r[2] * np.cos(theta)
        else:
            ## p0, p3
            # print poly
            theta = np.arctan2((poly[3][0] - poly[0][0]), (poly[3][1] - poly[0][1]))
            poly[0][0] += R * r[0] * np.sin(theta)
            poly[0][1] += R * r[0] * np.cos(theta)
            poly[3][0] -= R * r[3] * np.sin(theta)
            poly[3][1] -= R * r[3] * np.cos(theta)
            ## p1, p2
            theta = np.arctan2((poly[2][0] - poly[1][0]), (poly[2][1] - poly[1][1]))
            poly[1][0] += R * r[1] * np.sin(theta)
            poly[1][1] += R * r[1] * np.cos(theta)
            poly[2][0] -= R * r[2] * np.sin(theta)
            poly[2][1] -= R * r[2] * np.cos(theta)
            ## p0, p1
            theta = np.arctan2((poly[1][1] - poly[0][1]), (poly[1][0] - poly[0][0]))
            poly[0][0] += R * r[0] * np.cos(theta)
            poly[0][1] += R * r[0] * np.sin(theta)
            poly[1][0] -= R * r[1] * np.cos(theta)
            poly[1][1] -= R * r[1] * np.sin(theta)
            ## p2, p3
            theta = np.arctan2((poly[2][1] - poly[3][1]), (poly[2][0] - poly[3][0]))
            poly[3][0] += R * r[3] * np.cos(theta)
            poly[3][1] += R * r[3] * np.sin(theta)
            poly[2][0] -= R * r[2] * np.cos(theta)
            poly[2][1] -= R * r[2] * np.sin(theta)
        return poly

    @staticmethod
    def __fit_line(p1, p2):
        # fit a line ax+by+c = 0
        if p1[0] == p1[1]:
            return [1., 0., -p1[0]]
        else:
            [k, b] = np.polyfit(p1, p2, deg=1)
            return [k, -1., b]

    @staticmethod
    def __point_dist_to_line(p1, p2, p3):
        # compute the distance from p3 to p1-p2
        distance = 0
        try:
            eps = 1e-5
            distance = np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / (np.linalg.norm(p2 - p1) + eps)

        except:
            print('point dist to line raise Exception')

        return distance

    @staticmethod
    def __line_cross_point(line1, line2):
        # line1 0= ax+by+c, compute the cross point of line1 and line2
        if line1[0] != 0 and line1[0] == line2[0]:
            print('Cross point does not exist')
            return None
        if line1[0] == 0 and line2[0] == 0:
            print('Cross point does not exist')
            return None
        if line1[1] == 0:
            x = -line1[2]
            y = line2[0] * x + line2[2]
        elif line2[1] == 0:
            x = -line2[2]
            y = line1[0] * x + line1[2]
        else:
            k1, _, b1 = line1
            k2, _, b2 = line2
            x = -(b1 - b2) / (k1 - k2)
            y = k1 * x + b1
        return np.array([x, y], dtype=np.float32)

    def __rectangle_from_parallelogram(self, poly):
        '''
        fit a rectangle from a parallelogram
        :param poly:
        :return:
        '''
        p0, p1, p2, p3 = poly
        assert (np.linalg.norm(p0 - p1) * np.linalg.norm(p3 - p0)) != 0
        angle_p0 = np.arccos(np.dot(p1 - p0, p3 - p0) / (np.linalg.norm(p0 - p1) * np.linalg.norm(p3 - p0)))
        if angle_p0 < 0.5 * np.pi:
            if np.linalg.norm(p0 - p1) > np.linalg.norm(p0 - p3):
                # p0 and p2
                ## p0
                p2p3 = self.__fit_line([p2[0], p3[0]], [p2[1], p3[1]])
                p2p3_verticle = self.__line_verticle(p2p3, p0)

                new_p3 = self.__line_cross_point(p2p3, p2p3_verticle)
                ## p2
                p0p1 = self.__fit_line([p0[0], p1[0]], [p0[1], p1[1]])
                p0p1_verticle = self.__line_verticle(p0p1, p2)

                new_p1 = self.__line_cross_point(p0p1, p0p1_verticle)
                return np.array([p0, new_p1, p2, new_p3], dtype=np.float32)
            else:
                p1p2 = self.__fit_line([p1[0], p2[0]], [p1[1], p2[1]])
                p1p2_verticle = self.__line_verticle(p1p2, p0)

                new_p1 = self.__line_cross_point(p1p2, p1p2_verticle)
                p0p3 = self.__fit_line([p0[0], p3[0]], [p0[1], p3[1]])
                p0p3_verticle = self.__line_verticle(p0p3, p2)

                new_p3 = self.__line_cross_point(p0p3, p0p3_verticle)
                return np.array([p0, new_p1, p2, new_p3], dtype=np.float32)
        else:
            if np.linalg.norm(p0 - p1) > np.linalg.norm(p0 - p3):
                # p1 and p3
                ## p1
                p2p3 = self.__fit_line([p2[0], p3[0]], [p2[1], p3[1]])
                p2p3_verticle = self.__line_verticle(p2p3, p1)

                new_p2 = self.__line_cross_point(p2p3, p2p3_verticle)
                ## p3
                p0p1 = self.__fit_line([p0[0], p1[0]], [p0[1], p1[1]])
                p0p1_verticle = self.__line_verticle(p0p1, p3)

                new_p0 = self.__line_cross_point(p0p1, p0p1_verticle)
                return np.array([new_p0, p1, new_p2, p3], dtype=np.float32)
            else:
                p0p3 = self.__fit_line([p0[0], p3[0]], [p0[1], p3[1]])
                p0p3_verticle = self.__line_verticle(p0p3, p1)

                new_p0 = self.__line_cross_point(p0p3, p0p3_verticle)
                p1p2 = self.__fit_line([p1[0], p2[0]], [p1[1], p2[1]])
                p1p2_verticle = self.__line_verticle(p1p2, p3)

                new_p2 = self.__line_cross_point(p1p2, p1p2_verticle)
                return np.array([new_p0, p1, new_p2, p3], dtype=np.float32)

    @staticmethod
    def __line_verticle(line, point):
        # get the verticle line from line across point
        if line[1] == 0:
            verticle = [0, -1, point[1]]
        else:
            if line[0] == 0:
                verticle = [1, 0, -point[0]]
            else:
                verticle = [-1. / line[0], -1, point[1] - (-1 / line[0] * point[0])]
        return verticle

    @staticmethod
    def __line_cross_point(line1, line2):
        # line1 0= ax+by+c, compute the cross point of line1 and line2
        if line1[0] != 0 and line1[0] == line2[0]:
            print('Cross point does not exist')
            return None
        if line1[0] == 0 and line2[0] == 0:
            print('Cross point does not exist')
            return None
        if line1[1] == 0:
            x = -line1[2]
            y = line2[0] * x + line2[2]
        elif line2[1] == 0:
            x = -line2[2]
            y = line1[0] * x + line1[2]
        else:
            k1, _, b1 = line1
            k2, _, b2 = line2
            x = -(b1 - b2) / (k1 - k2)
            y = k1 * x + b1
        return np.array([x, y], dtype=np.float32)

    @staticmethod
    def __sort_rectangle(poly):
        # sort the four coordinates of the polygon, points in poly should be sorted clockwise
        # First find the lowest point
        p_lowest = np.argmax(poly[:, 1])
        if np.count_nonzero(poly[:, 1] == poly[p_lowest, 1]) == 2:
            # 底边平行于X轴, 那么p0为左上角
            p0_index = np.argmin(np.sum(poly, axis=1))
            p1_index = (p0_index + 1) % 4
            p2_index = (p0_index + 2) % 4
            p3_index = (p0_index + 3) % 4
            return poly[[p0_index, p1_index, p2_index, p3_index]], 0.
        else:
            # 找到最低点右边的点
            p_lowest_right = (p_lowest - 1) % 4
            p_lowest_left = (p_lowest + 1) % 4
            angle = np.arctan(
                -(poly[p_lowest][1] - poly[p_lowest_right][1]) / (poly[p_lowest][0] - poly[p_lowest_right][0]))
            # assert angle > 0
            # if angle <= 0:
            # print(angle, poly[p_lowest], poly[p_lowest_right])
            if angle / np.pi * 180 > 45:
                # 这个点为p2
                p2_index = p_lowest
                p1_index = (p2_index - 1) % 4
                p0_index = (p2_index - 2) % 4
                p3_index = (p2_index + 1) % 4
                return poly[[p0_index, p1_index, p2_index, p3_index]], -(np.pi / 2 - angle)
            else:
                # 这个点为p3
                p3_index = p_lowest
                p0_index = (p3_index + 1) % 4
                p1_index = (p3_index + 2) % 4
                p2_index = (p3_index + 3) % 4
                return poly[[p0_index, p1_index, p2_index, p3_index]], angle


class PseDataset(data.Dataset):
    def __init__(self, **kwargs):
        self.folder = get_valid('folder', kwargs)
        self.transforms = get('transforms', kwargs, None)
        self.input_size = get('input_size', kwargs, (768, 768))
        self.detection_transforms = get('detection_transforms', kwargs, None)
        self.data_list = self.__get_data_list()
        self.n = get('n', kwargs, 6)
        self.m = get('m', kwargs, 0.5)

    def __get_data_list(self):
        assert os.path.exists(self.folder)
        img_list = get_img_list(self.folder)
        data_list = list()
        for img in img_list:
            data = dict()
            img_path = os.path.join(self.folder, img)
            img_name = os.path.splitext(img)[0]
            txt_file = img_name + '.txt'
            txt_path = os.path.join(self.folder, txt_file)
            if not os.path.exists(txt_path):
                raise RuntimeError('{} not exists!'.format(txt_path))
            else:
                data['img_path'] = img_path
                data['txt_path'] = txt_path
                data_list.append(data)
        return data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        status = True
        while status:
            try:
                data = self.data_list[idx]
                img, score_maps, training_mask, img_path = self.__get_label(data)
                status = False
            except BaseException as e:
                print('Error on get label in ModelHelper.Detection.DetectionModels.Dataset: {}'.format(e))
            if status is True:
                idx = random.randint(0, len(self.data_list))
        img = img.astype(np.float32)
        img = self.transforms(img)
        output_data = dict()
        output_data['img'] = img
        output_data['score_maps'] = score_maps
        output_data['training_mask'] = training_mask
        output_data['img_path'] = img_path
        return output_data

    def __get_label(self, data):
        img_path = data['img_path']
        txt_path = data['txt_path']

        img = cv2.imread(img_path)
        h, w, _ = img.shape
        text_polys, text_tags = self.__load_annoataion(txt_path)
        text_polys, text_tags = self.__check_and_validate_polys(text_polys, text_tags, (h, w))

        if self.detection_transforms is not None:
            img, text_polys = self.detection_transforms(img, text_polys)

        new_h, new_w, _ = img.shape
        resize_h = self.input_size[0]
        resize_w = self.input_size[1]
        # resize the image to input size
        img = cv2.resize(img, dsize=(resize_w, resize_h))

        if text_polys.shape[0] == 0:
            score_maps = np.zeros((resize_h, resize_w, self.n))
            training_mask = np.ones((resize_h, resize_w), dtype=np.uint8)
        else:
            resize_ratio_3_x = resize_w / float(new_w)
            resize_ratio_3_y = resize_h / float(new_h)

            text_polys[:, :, 0] *= resize_ratio_3_x
            text_polys[:, :, 1] *= resize_ratio_3_y

            training_mask = np.ones((resize_h, resize_w), dtype=np.uint8)
            score_maps = []
            for i in range(1, self.n + 1):
                # s1->sn,由小到大
                score_map, training_mask = self.__generate_rbox((resize_h, resize_w), text_polys, text_tags,
                                                                training_mask, i, self.n, self.m)
                score_maps.append(score_map)
            score_maps = np.array(score_maps, dtype=np.float32)
        return img, score_maps, training_mask, img_path

    @staticmethod
    def __generate_rbox(im_size, text_polys, text_tags, training_mask, i, n, m):
        """
        生成mask图，白色部分是文本，黑色是北京
        :param im_size: 图像的h,w
        :param text_polys: 框的坐标
        :param text_tags: 标注文本框是否参与训练
        :return: 生成的mask图
        """
        h, w = im_size
        score_map = np.zeros((h, w), dtype=np.uint8)
        for poly, tag in zip(text_polys, text_tags):
            poly = poly.astype(np.int)
            r_i = 1 - (1 - m) * (n - i) / (n - 1)
            d_i = cv2.contourArea(poly) * (1 - r_i * r_i) / cv2.arcLength(poly, True)
            pco = pyclipper.PyclipperOffset()
            # pco.AddPath(pyclipper.scale_to_clipper(poly), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
            # shrinked_poly = np.floor(np.array(pyclipper.scale_from_clipper(pco.Execute(-d_i)))).astype(np.int)
            pco.AddPath(poly, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
            shrinked_poly = np.array(pco.Execute(-d_i))
            cv2.fillPoly(score_map, shrinked_poly, 1)
            # 制作mask
            # rect = cv2.minAreaRect(shrinked_poly)
            # poly_h, poly_w = rect[1]

            # if min(poly_h, poly_w) < 10:
            #     cv2.fillPoly(training_mask, shrinked_poly, 0)
            if tag:
                cv2.fillPoly(training_mask, shrinked_poly, 0)
            # 闭运算填充内部小框
            # kernel = np.ones((3, 3), np.uint8)
            # score_map = cv2.morphologyEx(score_map, cv2.MORPH_CLOSE, kernel)
        return score_map, training_mask

    def __check_and_validate_polys(self, polys, tags, xxx_todo_changeme):
        '''
        check so that the text poly is in the same direction,
        and also filter some invalid polygons
        :param polys:
        :param tags:
        :return:
        '''

        (h, w) = xxx_todo_changeme
        if polys.shape[0] == 0:
            return polys
        polys[:, :, 0] = np.clip(polys[:, :, 0], 0, w - 1)
        polys[:, :, 1] = np.clip(polys[:, :, 1], 0, h - 1)

        validated_polys = []
        validated_tags = []

        # find top-left and clockwise
        polys = self.__choose_best_begin_point(polys)

        for poly, tag in zip(polys, tags):
            p_area = self.__polygon_area(poly)
            if abs(p_area) < 1:
                # print poly
                # print('invalid poly')
                continue
            if p_area > 0:
                # print('poly in wrong direction')
                poly = poly[(0, 3, 2, 1), :]
            validated_polys.append(poly)
            validated_tags.append(tag)
        return np.array(validated_polys), np.array(validated_tags)

    @staticmethod
    def __load_annoataion(txt_path):
        '''
        load annotation from the text file

        Note:
        modified
        1. top left vertice
        2. clockwise

        :param p:
        :return:
        '''
        text_polys = []
        text_tags = []
        if not os.path.exists(txt_path):
            return np.array(text_polys, dtype=np.float32)
        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.encode('utf-8').decode('utf-8-sig')
                splited = line.strip().split(',')
                x1, y1 = splited[0], splited[1]
                x2, y2 = splited[2], splited[3]
                x3, y3 = splited[4], splited[5]
                x4, y4 = splited[6], splited[7]

                text_polys.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
                label = 'label'
                if label == '*' or label == '###':
                    text_tags.append(True)
                else:
                    text_tags.append(False)
        return np.array(text_polys, dtype=np.float32), np.array(text_tags, dtype=np.bool)

    def __choose_best_begin_point(self, pre_result):
        """
        find top-left vertice and resort
        """
        final_result = []
        for coordinate in pre_result:
            x1 = coordinate[0][0]
            y1 = coordinate[0][1]
            x2 = coordinate[1][0]
            y2 = coordinate[1][1]
            x3 = coordinate[2][0]
            y3 = coordinate[2][1]
            x4 = coordinate[3][0]
            y4 = coordinate[3][1]
            xmin = min(x1, x2, x3, x4)
            ymin = min(y1, y2, y3, y4)
            xmax = max(x1, x2, x3, x4)
            ymax = max(y1, y2, y3, y4)
            combinate = [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
                         [[x2, y2], [x3, y3], [x4, y4], [x1, y1]],
                         [[x3, y3], [x4, y4], [x1, y1], [x2, y2]],
                         [[x4, y4], [x1, y1], [x2, y2], [x3, y3]]]
            dst_coordinate = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
            force = 100000000.0
            force_flag = 0
            for i in range(4):
                temp_force = self.__calculate_distance(combinate[i][0], dst_coordinate[0]) + self.__calculate_distance(
                    combinate[i][1], dst_coordinate[1]) + self.__calculate_distance(combinate[i][2], dst_coordinate[
                    2]) + self.__calculate_distance(combinate[i][3], dst_coordinate[3])
                if temp_force < force:
                    force = temp_force
                    force_flag = i
            # if force_flag != 0:
            #    print("choose one direction!")
            final_result.append(combinate[force_flag])

        return final_result

    def __polygon_area(slef, poly):
        '''
        compute area of a polygon
        :param poly:
        :return:
        '''
        poly_ = np.array(poly)
        assert poly_.shape == (4, 2), 'poly shape should be 4,2'
        edge = [
            (poly[1][0] - poly[0][0]) * (poly[1][1] + poly[0][1]),
            (poly[2][0] - poly[1][0]) * (poly[2][1] + poly[1][1]),
            (poly[3][0] - poly[2][0]) * (poly[3][1] + poly[2][1]),
            (poly[0][0] - poly[3][0]) * (poly[0][1] + poly[3][1])
        ]
        return np.sum(edge) / 2.

    @staticmethod
    def __calculate_distance(c1, c2):
        return math.sqrt(math.pow(c1[0] - c2[0], 2) + math.pow(c1[1] - c2[1], 2))
