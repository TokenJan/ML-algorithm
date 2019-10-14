from math import log

class DecisionTree(object):
    def __init__(self, max_depth):
        self.max_depth = max_depth
        self.current_depth = 0

    '''计算信息熵'''
    def calcEntropy(self, data):
        numEntities = len(data) # 样本数D
        labelCount = {} # key存放类别 value存放对应类别个数
        for line in data:
            currentLabel = line[-1]
            if currentLabel not in labelCount.keys():  #不包含此类别则加入此类别
                labelCount[currentLabel] = 0
            labelCount[currentLabel] += 1
        shannonEnt = 0.0 #信息熵
        for key in labelCount:
            prob = float(labelCount[key])/numEntities
            shannonEnt -= prob * log(prob,2)
    #         print(prob)
        return shannonEnt

    '''根据指定特征和特征值划分数据'''
    def split_DataSet(self, data, axis, value):
        res_data = []
        for vec in data:
            if vec[axis]==value:
                reducedFeatVec = vec[:axis]
                reducedFeatVec.extend(vec[axis+1:])
    #             print(reducedFeatVec)
                res_data.append(reducedFeatVec)
        return res_data

    '''选择最优信息特征'''
    def choose_best_feature(self, data):
        feature_size = len(data[0]) - 1
        base_entropy = self.calcEntropy(data)
        best_Gain = 0.0
        best_feature = -1
        for i in range(feature_size):#遍历特征
            feat_list = [temp[i] for temp in data]
    #         print(feat_list)
            feat_set = set(feat_list)
            new_entropy = 0.0
            for feat_value in feat_set:
                sub_dataset = self.split_DataSet(data,i,feat_value)
                prob = len(sub_dataset)/float(len(data))
                new_entropy += prob * self.calcEntropy(sub_dataset)
            info_gain = base_entropy - new_entropy
            print(info_gain)
            if(info_gain > best_Gain):
                best_Gain = info_gain
                best_feature = i
        return best_feature

    '''分类，创建字典，排序并返回出现次数最多的分类名称'''
    def majority_cnt(self, class_list):
        class_count = {}
        for vote in class_list:
            if vote not in class_count.keys():class_count[vote] = 0
            class_count[vote] += 1
        sorted_class_count = sorted(class_count.iteritems(), key=operator.itemgetter(1), reverse=True)
        return sorted_class_count[0][0]

    '''生成决策树主方法'''
    def create_tree(self, data, labels):
        class_list = [temp[-1] for temp in data]
        # if len(class_list) == 0:
        #     return "不嫁"
        if class_list.count(class_list[0]) == len(class_list):
            return class_list[0]
        if len(class_list) == 1:
            return self.majority_cnt(class_list)
        best_feature = self.choose_best_feature(data)
        best_feat_label = labels[best_feature]
        
        mytree = {best_feat_label:{}}
        del(labels[best_feature])
        feat_values = [temp[best_feature] for temp in data]
        unique_vals = set(feat_values)
        for value in unique_vals:
            sub_label = labels[:]
            mytree[best_feat_label][value] = self.create_tree(self.split_DataSet(data,best_feature,value),sub_label)
        return mytree

def load_data():
    data = [
            ["帅", "不好", "矮", "不上进", "不嫁"],
            ["不帅", "好", "矮", "上进", "不嫁"],
            ["帅", "好", "矮", "上进", "嫁"],
            ["不帅", "很好", "高", "上进", "嫁"],
            ["帅", "不好", "矮", "上进", "不嫁"],
            ["帅", "不好", "矮", "上进", "不嫁"],
            ["帅", "好", "高", "不上进", "嫁"],
            ["不帅", "好", "中", "上进", "嫁"],
            ["帅", "很好", "中", "上进", "嫁"],
            ["不帅", "不好", "高", "上进", "嫁"],
            ["帅", "好", "矮", "不上进", "不嫁"],
            ["帅", "好", "矮", "不上进", "不嫁"]
        ]
    labels = ["颜值", "性格", "身高", "事业"]
    return data, labels


if __name__ == "__main__":
    data, labels = load_data()
    model = DecisionTree(1)
    print(model.create_tree(data, labels))