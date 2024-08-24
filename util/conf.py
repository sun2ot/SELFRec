import os.path


class ModelConf(object):
    """
    配置文件读取类，产生一个 dict
    """
    def __init__(self,file):
        self.config = {}
        self.read_configuration(file)

    def __getitem__(self, item):
        """获取配置项"""
        if not self.contain(item):
            print('parameter '+item+' is not found in the configuration file!')
            exit(-1)
        return self.config[item]

    def contain(self, key):
        """判断配置中是否包含指定的键"""
        return key in self.config

    def read_configuration(self,file):
        if not os.path.exists(file):
            print('config file is not found!')
            raise IOError
        with open(file) as f:
            # 逐行读取配置文件
            for ind,line in enumerate(f):
                if line.strip()!='':
                    try:
                        #! 配置文件谨慎空格
                        key,value=line.strip().split('=')
                        self.config[key]=value
                    except ValueError:
                        print('config file is not in the correct format! Error Line:%d' % ind)


class OptionConf(object):
    def __init__(self,content):
        """
        分解参数选项，如
        `-topN 10,20` -> {'-topN':'10,20'}
        """
        self.line = content.strip().split(' ')
        self.options = {}
        self.mainOption = False

        if self.line[0] == 'on':
            self.mainOption = True
        elif self.line[0] == 'off':
            self.mainOption = False
        
        for i, item in enumerate(self.line):
            if (item.startswith('-') or item.startswith('--')) and not item[1:].isdigit():
                ind = i+1
                for j,sub in enumerate(self.line[ind:]):
                    if (sub.startswith('-') or sub.startswith('--')) and not sub[1:].isdigit():
                        ind = j
                        break
                    if j == len(self.line[ind:])-1:
                        ind=j+1
                        break
                try:
                    self.options[item] = ' '.join(self.line[i+1:i+1+ind])
                except IndexError:
                    self.options[item] = 1

    def __getitem__(self, item):
        if not self.contain(item):
            print('parameter '+item+' is invalid!')
            exit(-1)
        return self.options[item]

    def keys(self):
        return self.options.keys()

    def is_main_on(self):
        return self.mainOption

    def contain(self,key):
        return key in self.options


