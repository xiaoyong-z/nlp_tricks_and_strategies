# -*- coding: utf-8 -*-

import os
from configparser import ConfigParser


def get_conf(path):
	conf = ConfigParser()
	conf.read(path, encoding="UTF-8")
	return conf


class Dict(dict):
	"""
	重写 dict, 使之通过 “.” 调用
	带参数 key 的 __call__ 方法用于实例自身的调用, 达到 () 调用的效果
	"""
	def __init__(self, *args, **kwargs):
		super(Dict, self).__init__(*args, **kwargs)

	def __getattr__(self, key):
		try:
			value = self[key]
			if isinstance(value, dict):
				value = Dict(value)
			return value
		except KeyError as k:
			return None

	def __setattr__(self, key, value):
		if isinstance(value, dict):
			value = Dict(value)
		self[key] = value

	def __delattr__(self, key):
		try:
			del self[key]
		except KeyError as k:
			return None

	def __call__(self, key):
		try:
			return self[key]
		except KeyError as k:
			return None


class Config(object):
	"""链式配置
	"""
	def __init__(self, filepath=None):
		if filepath:
			self.path = filepath
		else:
			cur_dir = os.path.split(os.path.realpath(__file__))[0]
			self.path = os.path.join(cur_dir, "conf", "self.conf")
		self.conf = get_conf(self.path)
		self.d = Dict()
		for s in self.conf.sections():
			value = Dict()
			for k in self.conf.options(s):
				value[k] = self.conf.get(s, k)
			self.d[s] = value

	def add(self, section):
		self.conf.add_section(section)
		self.d[section] = Dict()
		with open(self.path, 'w', encoding="UTF-8") as f:
			self.conf.write(f)

	def set(self, section, k, v):
		self.conf.set(section, k, v)
		self.d[section][k] = v
		with open(self.path, 'w', encoding="UTF-8") as f:
			self.conf.write(f)

	def get(self, section, key):
		return self.conf.get(section, key, default=None)
		# return self.d[section][k]

	def remove_section(self, section):
		self.conf.remove_section(section)
		del self.d[section]
		with open(self.path, 'w', encoding="UTF-8") as f:
			self.conf.write(f)

	def remove_option(self, section, key):
		self.conf.remove_option(section, key)
		del self.d[section][key]
		with open(self.path, 'w', encoding="UTF-8") as f:
			self.conf.write(f)

	def save(self):
		for s in self.d:
			if s not in self.conf.sections():
				self.add(s)
			for k in self.d[s]:
				try:
					v = self.get(s, k)
				except:
					v = None
				if self.d[s][k] != v:
					self.set(s, k, self.d[s][k])

	def __getattr__(self, name):
		if name not in self.__dict__:
			try:
				return self.d[name]
			except KeyError as k:
				self.d[name] = Dict()
				return self.d[name]
		return self.__dict__[name]


if __name__ == '__main__':
	c = Config()

	# print(c.segment.tool)
	# c.segment.tool = "jieba"
	# print(c.segment.tool)

	# print(c.segment.embedding)
	# c.segment.embedding = "word2vec"
	# print(c.segment.embedding)

	# print(c.back)
	# c.back.tf = "1.10.0"
	# print(c.back)

	# c.remove_section("back")
	# c.remove_option("segment", "embedding")

	c.save()