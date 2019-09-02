from glob import glob
from math import ceil
import multiprocessing as mp
import os
import re

import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

import numpy as  np
import torch
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm
import random
import seaborn as sns

class Batch(object):
    def __init__(self):
        self.__doc__ = "empty initialization"

    @classmethod
    def get_batch(cls, keys=None, values=None):
        """Create a Batch directly from a number of Variables."""
        batch = cls()
        assert keys is not None and values is not None
        for k, v in zip(keys, values):
            setattr(batch, k, v)
        return batch

class MemoryEntry(object):
    def __init__(self, data, idx=-1):
        # For video environment, make data as tuple of (video, sub)
        self.data = data
        self.feature = None
        self.value = None
        self.hidden = None
        self.idx = idx

    def get_data(self):
        return self.data

    def check_feature(self):
        return self.feature is not None



class Environment(object):
    def __init__(self, cfg, set_id, dset, shuffle):
        self.cfg = cfg
        # Dataset Feature (with memory entry)
        self.dset = dset
        self.shuffle = shuffle
        # used data index of this episode
        self.data_idx = 0

    def set_model(self, model):
        self._model = model

    def set_gpu_id(self, gpu_id):
        self._gpu_id = gpu_id

    def reset(self, idx=-1):
        if idx < 0:
            if self.shuffle:
                self.data_idx = np.random.randint(len(self.dset))
            else:
                self.data_idx = (self.data_idx + 1) % len(self.dset)
        else:
            self.data_idx = idx

        if self.cfg.debug:
            self.data_idx = 0

        self.data = self.dset[self.data_idx]
        self.memory = []
        # pointer indicates data(waiting memory entry) index
        self.data_ptr = 0

        # Reset includes fill up memory
        while len(self.memory) < self.cfg.memory_num:
            if self.data_ptr >= len(self.data[-2]):
                self.memory.append(MemoryEntry(None))
            else:
                if self.data_ptr >= len(self.data[-1]):
                    sub = None
                else:
                    sub = self.data[-1][self.data_ptr]
                self.memory.append(MemoryEntry((self.data[-2][self.data_ptr], sub), self.data_ptr))
                # print("DATA IDX : %d" % self.data[self.data_ptr].memory_index)
                self.data_ptr += 1

        assert len(self.memory) == self.cfg.memory_num


    def observe(self):
        solvable = True
        return self.memory, solvable

    def step(self, idx):
        if self.cfg.model == "FIFO":
            idx = 0
        elif self.cfg.model == "LIFO":
            idx = self.cfg.memory_num - 1
        elif self.cfg.model == "UNIFORM":
            idx = random.randint(0, self.cfg.memory_num-1)

        self.memory.pop(idx)
        return

    def step_append(self):
        # append next sentence
        if self.data_ptr >= len(self.data[-1]):
            sub = None
        else:
            sub = self.data[-1][self.data_ptr]
        self.memory.append(MemoryEntry((self.data[-2][self.data_ptr], sub), self.data_ptr))
        self.data_ptr += 1

        assert len(self.memory) == self.cfg.memory_num
        return


    def qa_construct(self, device):
        # This function returns full context which
        # is constructed based on memory state
        data = self.data
        text_keys = ["q", "a0", "a1", "a2", "a3", "a4", "sub", "vcpt"]
        sub_key = "sub_new"
        label_key = "answer_idx"
        qid_key = "qid"
        vid_name_key = "vid_name"
        vid_feat_key = "vid"
        all_keys = text_keys + [label_key, qid_key, vid_name_key, vid_feat_key, sub_key]
        all_values = []

        realloc_memory = []
        for entry in self.memory:
            if entry.data is not None:
                realloc_memory.append(entry)

        vid_feature = torch.stack([entry.data[0] for entry in realloc_memory], 0)
        sub_feature = []
        for entry in realloc_memory:
            if entry.data[1] is not None:
                sub_feature.extend(entry.data[1])

        if len(sub_feature) == 0:
            sub_feature = [0]

        for i, k in enumerate(all_keys):
            if k in text_keys:
                all_values.append([torch.LongTensor(data[i]), torch.LongTensor([len(data[i])])])
            elif k == label_key:
                all_values.append(torch.LongTensor([data[i]]))
            elif k == vid_feat_key:
                all_values.append([vid_feature, torch.LongTensor([len(self.memory)])])
            elif k == sub_key:
                all_values.append([torch.LongTensor(sub_feature), torch.LongTensor([len(sub_feature)])])
            else:
                all_values.append(data[i])

        batch = Batch.get_batch(keys=all_keys, values=all_values)

        max_len_dict = {"sub":300, "vcpt":300, "sub_new":300}
        text_keys = ["q", "a0", "a1", "a2", "a3", "a4", "sub_new", "vcpt"]
        label_key = "answer_idx"
        qid_key = "qid"
        vid_feat_key = "vid"
        model_in_list = []
        for k in text_keys + [vid_feat_key]:
            v = getattr(batch, k)
            if k in max_len_dict:
                ctx, ctx_l = v
                max_l = min(ctx.size(0), max_len_dict[k])
                if ctx.size(0) > max_l:
                    ctx_l = ctx_l.clamp(min=1, max=max_l)
                    ctx = ctx[:max_l]
                model_in_list.extend([ctx.cuda(device).unsqueeze(0), ctx_l.cuda(device).unsqueeze(0)])
            else:
                model_in_list.extend([v[0].cuda(device).unsqueeze(0), v[1].cuda(device).unsqueeze(0)])
        target_data = getattr(batch, label_key)
        target_data = target_data.cuda(device)
        qid_data = getattr(batch, qid_key)
        return model_in_list, target_data, qid_data


    def is_done(self):
        # print(self.data_ptr)
        return self.data_ptr >= len(self.data[-2])

    def invest_memory(self):
        data = self.dset.get_original_item(self.data_idx)
        frame_locate = data["located_frame"]
        solvable = 0
        for i, memory in enumerate(self.memory):
            if frame_locate[0] <= memory.idx and memory.idx <= frame_locate[1]:
                solvable += 1

        return solvable / len(self.memory)

    def print_memory(self, pic_path, prob=None, answer_set=None, att=None):
        _, solvable = self.observe()
        print(" === This memory is %s ===\n" % ('Solvable' if solvable else 'Unsolvable') )
        data = self.dset.get_original_item(self.data_idx)
        frame_locate = data["located_frame"]
        show_name = data["show_name"]
        cur_vid_name = data["vid_name"]
        subtitle = data["subtitle"]
        SHOW2PATH = {"The Big Bang Theory":"bbt_frames", "Castle":"castle_frames", "House M.D.":"house_frames",
                     "Friends":"friends_frames", "Grey's Anatomy":"grey_frames", "How I Met You Mother":"met_frames"}
        pic_path = os.path.join(pic_path, SHOW2PATH[show_name], cur_vid_name)
        print(pic_path)
        # Remeber! Plus 1 to index (in path, it starts with number 00001
        sample = Image.open(os.path.join(pic_path, "1".zfill(5) + ".jpg"))
        im_size = sample.size
        # In case of 20 memory cells
        # album = Image.new("RGB", (im_size[0] * (self.cfg.memory_num // 2), int(im_size[1] * 6.5)))
        album = Image.new("RGB", (im_size[0] * (10), int(im_size[1] * (2.5 + self.cfg.memory_num // 10 * 2))))
        draw = ImageDraw.Draw(album)

        # Font
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        font_size = 70
        font = ImageFont.truetype(font_path, font_size)

        font_bold_path = "/usr/share/fonts/truetype/freefont/DejaVuSans-Bold.ttf"
        font_bold = ImageFont.truetype(font_bold_path, font_size)
        ####

        locate_check = ""
        if prob is not None:
            max_prob = torch.max(prob, 1)[1].item()

        for i, memory in enumerate(self.memory):
            height = (i // 10) * 2
            width = i % 10
            idx = str(memory.idx + 1)
            scene = Image.open(os.path.join(pic_path, idx.zfill(5) + ".jpg"))
            album.paste(scene, (width * im_size[0], im_size[1] * height))
            if memory.data[1] is not None:
                draw.text(((width + 0.3) * im_size[0], im_size[1] * (1.1 + height)), "SUBTITLE", fill='blue', font=font)

            # Visualize Delete prob
            if prob is not None:
                if i < len(prob[0]):
                    ret = Image.new("RGB", im_size, (int(255 * prob[0][i].item()),)*3)
                    album.paste(ret, (width * im_size[0], im_size[1] * (1 + height)))

                    f = font_bold if i == max_prob else font
                    draw.text(((width + 0.3) * im_size[0], im_size[1] * (1.4 + height)), '%.3f' % (prob[0][i].item()), fill='red', font=f)

                # Draw white box on max prob
                if i == max_prob:
                    x1 = width * im_size[0] + 2
                    y1 = im_size[1] * (1 + height) + 5
                    x2 = (width + 1) * im_size[0] - 2
                    y2 = im_size[1] * (2 + height)
                    points = (x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)
                    draw.line(points, fill='white', width=10)

            # Print Subtitle
            if memory.idx < len(subtitle) and subtitle[memory.idx] is not None:
                print("%dth Scene's SUBTITLE : %s" % (i, subtitle[memory.idx]))


            # Draw Red box on supporting frames
            if frame_locate[0]+1 <= int(idx) and int(idx) <= frame_locate[1]+1:
                x1 = width * im_size[0]
                y1 = im_size[1] * height + 7
                x2 = (width + 1) * im_size[0] - 12
                y2 = im_size[1] * (height + 1)
                points = (x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)
                draw.line(points, fill='red', width=5)
                locate_check += (str(memory.idx) + "O,")
            else:
                locate_check += (str(memory.idx) + "X,")

        draw.text((0.5 * im_size[0], int(im_size[1] * (0.1 + self.cfg.memory_num // 10 * 2))), "Question : " + data["q"], fill='white', font=font)
        print()
        for i in range(5):
            key = "a" + str(i)
            f = font
            color = 'white'
            if answer_set is not None:
                if i == answer_set[0]:
                    f = font_bold
                if i == answer_set[1]:
                    color = 'red'

            draw.text((0.5 * im_size[0], int(im_size[1] * (0.7 + self.cfg.memory_num // 10 * 2 + 0.3*i))), "Answer%d : " % (i) + data[key], \
                      fill=color, font=f)

        print("Question: %s" % (data["q"]))
        print("Answer0 : %s" % (data["a0"]))
        print("Answer1 : %s" % (data["a1"]))
        print("Answer2 : %s" % (data["a2"]))
        print("Answer3 : %s" % (data["a3"]))
        print("Answer4 : %s" % (data["a4"]))
        print("Locate Check : %s" % locate_check)
        print(frame_locate)
        print(prob)
