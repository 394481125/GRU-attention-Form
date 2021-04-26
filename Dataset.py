import torch.utils.data as data
import numpy as np
import os
from PIL import Image
import torch

# 根据txt文档读取视频及参数
def read_data(lines):
    list_strs = lines.split(' ')
    num_frame = list_strs[-2]
    label = list_strs[-1]
    path = list_strs[0]
    for idx in range(1, len(list_strs)-2):
        path = path + ' ' + list_strs[idx]

    return [path, num_frame, label]

# 获得每个视频的参数
class Videolist_Parse(object):
	'''将每个video的参数保留下来包含[path,num_frames_label]'''
	def __init__(self, row):
		self.row = row
	@property
	def path(self):
		return self.row[0]
	@property
	def num_frames(self):
		return int(self.row[1])
	@property
	def label(self):
		return int(self.row[2])

# 获得Dataset数据类型的数据
class VideoDataset(data.Dataset):
	def __init__(self, root, list, transform, num_segments, num_frames, test_mode=False):
		self.transform = transform
		self.list = list
		self.root = root
		self.num_segments = num_segments
		self.num_frames = num_frames
		self.test_mode = test_mode

		self._parse_videolist()

	def __len__(self):
		return len(self.videolist)

	# 返回图像的tensor和标签
	def __getitem__(self, idx):
		record = self.videolist[idx]

		if not self.test_mode:
			indices = self.get_indices(record)
			image_tensor = self.get_img(indices, record)
		else:
			image_tensor = []
			for count in range(10):
				indices = self.get_indices(record)
				image_tensor.append(self.get_img(indices, record))
			image_tensor = torch.stack(image_tensor, dim=0)
		return image_tensor, record.label

	# 获得所有视频的参数列表
	def _parse_videolist(self):
		'''获得video列表包含每个video的参数[path,num_frames_label]
			保存在videolist中
		'''
		lines = [read_data(x.strip()) for x in open(self.root + self.list)]
		self.videolist = [Videolist_Parse(item) for item in lines]

	def get_indices(self, record):
		# average_duration表示某个视频分成self.num_segments份的时候每一份包含多少帧图像
		average_duration = record.num_frames // self.num_segments
		# 生成了self.num_segments个范围在0到average_duration的数值，二者相加就相当于在这self.num_segments个片段中分别随机选择了一帧图像。

		choices = [np.random.choice(average_duration, self.num_frames, replace=False)+ i * average_duration for i in range(self.num_segments)]
		choices = np.concatenate((choices))
		offsets = np.sort(choices)
		return offsets

	# 根据视频的indices获取视频帧图像
	def get_img(self, indices, record):
		frames = torch.zeros(self.num_segments * self.num_frames, 3, 224, 224)
		for idx, idx_img in enumerate(indices):
			dir_img = os.path.join(self.root, record.path, str(idx_img+1)+'.jpg')
			image = Image.open(dir_img).convert('RGB')
			frames[idx] = self.transform(image)
		return frames
