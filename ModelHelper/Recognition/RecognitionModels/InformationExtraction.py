from fuzzywuzzy import fuzz
import math


def is_chinese(char):
    if '\u4e00' <= char <= '\u9fff':
        return True
    return False


def is_english(char):
    if 'a' <= char <= 'z' or 'A' <= char <= 'Z':
        return True
    return False


def is_number(char):
    if '0' <= char <= '9':
        return True
    return False


class Patch:
    def __init__(self, id, pos, info):
        self.id = id
        self.pos = pos
        self.info = info
        self.left = min(self.pos[:, 0])
        self.right = max(self.pos[:, 0])
        self.top = min(self.pos[:, 1])
        self.bottom = max(self.pos[:, 1])
        self.center = ((self.left + self.right) / 2 + (self.top + self.bottom) / 2)
        self.left_center = (self.left, (self.top + self.bottom) / 2)
        self.right_center = (self.right, (self.top + self.bottom) / 2)

    def extract_cn(self):
        output = ''
        for char in self.info:
            if is_chinese(char):
                output += char
        return output

    def delete_cn(self):
        output = ''
        for char in self.info:
            if not is_chinese(char):
                output += char
        return output

    def extract_en(self):
        output = ''
        for char in self.info:
            if is_english(char):
                output += char
        return output

    def delete_en(self):
        output = ''
        for char in self.info:
            if not is_english(char):
                output += char
        return output

    def extract_num(self):
        output = ''
        for char in self.info:
            if is_number(char):
                output += char
        return output

    def delete_num(self):
        output = ''
        for char in self.info:
            if not is_number(char):
                output += char
        return output


class ExtractionTemplate:
    def __init__(self, patch_list, key_list):
        self.patch_list = patch_list
        self.key_list = key_list

    def match_key_list(self, threshold):
        key2patch_dict = dict()
        for key in self.key_list:
            match_info = dict()
            match_info['patch'] = None
            match_info['score'] = 0
            key2patch_dict[key] = match_info

        for patch in self.patch_list:
            for key in self.key_list:
                score = fuzz.ratio(key, patch.info)
                if score > threshold and score > key2patch_dict[key]['score']:
                    key2patch_dict[key]['patch'] = patch
                    key2patch_dict[key]['score'] = score
        return key2patch_dict

    @staticmethod
    def get_closest_patch(target_patch, patch_list):
        best_patch = None
        min_distance = 9999999999
        if target_patch is not None:
            target_point = target_patch.right_center

            for patch in patch_list:
                point = patch.left_center
                distance = math.sqrt(
                    (point[0] - target_point[0]) * (point[0] - target_point[0]) + 10 * (point[1] - target_point[1]) * (
                            point[1] - target_point[1]))
                if distance < min_distance:
                    min_distance = distance
                    best_patch = patch

        return best_patch, min_distance
