import numpy as np
import torchvision
# from matplotlib import pyplot as plt
# from mpmath.libmp import normalize1






class NumPyResNet9:
    def __init__(self, label, input_shape=(3, 32, 32), num_classes=10, lr = 0.001):
        # 初始化参数
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.lr = lr
        self.label = label
        self.pred = None

        # 初始化权重
        self.weights = self._initialize_weights()

    def _initialize_weights(self):
        """初始化所有层的权重"""
        weights = {}
        in_channels, height, width = self.input_shape

        # 初始卷积层 (7x7, 64 filters, stride=2)
        # 输出尺寸计算: ((H-F+2P)/S)+1
        # 假设padding=3保持尺寸，stride=2则输出为(32-7+6)/2+1=16
        out_channels = 64
        weights['conv1'] = np.random.randn(out_channels, in_channels, 3, 3).astype(np.float32) * 0.01
        weights['conv1_bias'] = np.zeros(out_channels).astype(np.float32)
        weights['batchnorm1'] = {'gamma': np.ones(out_channels).astype(np.float32), 'beta': np.zeros(out_channels).astype(np.float32)}

        # 第一个残差块 (64->128, stride=2)
        weights['res1_conv1'] = np.random.randn(128, 64, 3, 3).astype(np.float32) * 0.01
        weights['res1_conv1_bias'] = np.zeros(128).astype(np.float32)
        weights['res1_batchnorm1'] = {'gamma': np.ones(128).astype(np.float32), 'beta': np.zeros(128).astype(np.float32)}
        weights['res1_conv2'] = np.random.randn(128, 128, 3, 3).astype(np.float32) * 0.01
        weights['res1_conv2_bias'] = np.zeros(128).astype(np.float32)
        weights['res1_batchnorm2'] = {'gamma': np.ones(128).astype(np.float32), 'beta': np.zeros(128).astype(np.float32)}
        # 跳跃连接的1x1卷积
        weights['res1_shortcut'] = np.random.randn(128, 64, 1, 1).astype(np.float32) * 0.01
        weights['res1_shortcut_bias'] = np.zeros(128).astype(np.float32)
        weights['res1_shortcut_batchnorm'] = {'gamma': np.ones(128).astype(np.float32), 'beta': np.zeros(128).astype(np.float32)}

        # 第二个残差块 (128->128)
        weights['res2_conv1'] = np.random.randn(128, 128, 3, 3).astype(np.float32) * 0.01
        weights['res2_conv1_bias'] = np.zeros(128).astype(np.float32)
        weights['res2_batchnorm1'] = {'gamma': np.ones(128).astype(np.float32), 'beta': np.zeros(128).astype(np.float32)}
        weights['res2_conv2'] = np.random.randn(128, 128, 3, 3).astype(np.float32) * 0.01
        weights['res2_conv2_bias'] = np.zeros(128).astype(np.float32)
        weights['res2_batchnorm2'] = {'gamma': np.ones(128).astype(np.float32), 'beta': np.zeros(128).astype(np.float32)}

        # 第三个残差块 (128->256, stride=2)
        weights['res3_conv1'] = np.random.randn(256, 128, 3, 3).astype(np.float32) * 0.01
        weights['res3_conv1_bias'] = np.zeros(256).astype(np.float32)
        weights['res3_batchnorm1'] = {'gamma': np.ones(256).astype(np.float32), 'beta': np.zeros(256).astype(np.float32)}
        weights['res3_conv2'] = np.random.randn(256, 256, 3, 3).astype(np.float32) * 0.01
        weights['res3_conv2_bias'] = np.zeros(256).astype(np.float32)
        weights['res3_batchnorm2'] = {'gamma': np.ones(256).astype(np.float32), 'beta': np.zeros(256).astype(np.float32)}
        # 跳跃连接的1x1卷积
        weights['res3_shortcut'] = np.random.randn(256, 128, 1, 1).astype(np.float32) * 0.01
        weights['res3_shortcut_bias'] = np.zeros(256).astype(np.float32)
        weights['res3_shortcut_batchnorm'] = {'gamma': np.ones(256).astype(np.float32), 'beta': np.zeros(256).astype(np.float32)}

        # 第四个残差块 (256->256)
        weights['res4_conv1'] = np.random.randn(256, 256, 3, 3).astype(np.float32) * 0.01
        weights['res4_conv1_bias'] = np.zeros(256).astype(np.float32)
        weights['res4_batchnorm1'] = {'gamma': np.ones(256).astype(np.float32), 'beta': np.zeros(256).astype(np.float32)}
        weights['res4_conv2'] = np.random.randn(256, 256, 3, 3).astype(np.float32) * 0.01
        weights['res4_conv2_bias'] = np.zeros(256).astype(np.float32)
        weights['res4_batchnorm2'] = {'gamma': np.ones(256).astype(np.float32), 'beta': np.zeros(256).astype(np.float32)}

        # 全连接层
        # 假设经过全局平均池化后特征图为8x8（需要根据实际输入调整）
        # 这里简化处理，实际实现需要跟踪特征图尺寸
        weights['fc'] = np.random.randn(256 * 1 * 1, self.num_classes).astype(np.float32) * 0.01
        weights['fc_bias'] = np.zeros(self.num_classes)

        return weights

    def _im2col(self, x, filter_h, filter_w, stride=1, pad=1):
        """将输入矩阵转换为列（im2col操作）"""
        N, C, H, W = x.shape
        out_h = (H + 2 * pad - filter_h) // stride + 1
        out_w = (W + 2 * pad - filter_w) // stride + 1

        img = np.pad(x, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
        col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

        for y in range(filter_h):
            y_max = y + stride * out_h
            for x in range(filter_w):
                x_max = x + stride * out_w
                col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

        col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
        return col, out_h, out_w

    def _col2im(self, col, input_shape, filter_h, filter_w, stride=1, pad=0):
        """将列转换回图像（col2im操作）"""
        N, C, H, W = input_shape
        out_h = (H + 2 * pad - filter_h) // stride + 1
        out_w = (W + 2 * pad - filter_w) // stride + 1
        col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

        img = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))
        for y in range(filter_h):
            y_max = y + stride * out_h
            for x in range(filter_w):
                x_max = x + stride * out_w
                img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

        return img[:, :, pad:H + pad, pad:W + pad]

    def _conv2d(self, x, w, b, stride=1, pad=1):
        """2D卷积实现（使用im2col加速）"""
        filter_num, _, filter_h, filter_w = w.shape
        N, C, H, W = x.shape

        col, out_h, out_w = self._im2col(x, filter_h, filter_w, stride, pad)
        w_col = w.reshape(filter_num, -1)
        out = np.dot(col, w_col.T) + b

        out = out.reshape(N, out.shape[0] // N, -1)
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        return out

    def _batch_norm(self, x, n, eps=1e-5):
        """简化版批量归一化"""
        # 这里简化处理，实际BN需要跟踪运行时的均值和方差
        # 这里仅做缩放和平移
        mean = np.mean(x, axis=(0, 2, 3), keepdims=True)
        var = np.var(x, axis=(0, 2, 3), keepdims=True)
        x_normalized = (x - mean) / np.sqrt(var + eps)
        return n['gamma'].reshape(1, len(n['gamma']), 1, 1) * x_normalized + n['beta'].reshape(1, len(n['beta']), 1, 1)

    def _relu(self, x):
        """ReLU激活函数"""
        return np.maximum(0, x)

    def softmax(self, x):
        '''softmax'''
        x = np.exp(x)
        return x / np.sum(x, axis=1, keepdims=True)

    def _global_avg_pool(self, x):
        """全局平均池化"""
        return np.mean(x, axis=(2, 3))

    def _residual_block(self, x, prefix, stride=1):
        """残差块实现"""
        w1 = self.weights[f'{prefix}_conv1']
        b1 = self.weights[f'{prefix}_conv1_bias']
        w2 = self.weights[f'{prefix}_conv2']
        b2 = self.weights[f'{prefix}_conv2_bias']
        n1 = self.weights[f'{prefix}_batchnorm1']
        n2 = self.weights[f'{prefix}_batchnorm2']

        # 第一个卷积层
        out = self._conv2d(x, w1, b1, stride=stride)
        out = self._batch_norm(out, n1)
        out = self._relu(out)

        # 第二个卷积层
        out = self._conv2d(out, w2, b2, stride=1)
        out = self._batch_norm(out, n2)

        # 跳跃连接
        if stride != 1 or x.shape[1] != out.shape[1]:
            shortcut = self._conv2d(x,
                                    self.weights[f'{prefix}_shortcut'],
                                    self.weights[f'{prefix}_shortcut_bias'],
                                    stride=stride, pad=0)
            shortcut = self._batch_norm(shortcut, self.weights[f'{prefix}_shortcut_batchnorm'])
        else:
            shortcut = x

        # 残差相加
        out += shortcut
        out = self._relu(out)
        return out

    def forward(self, x):
        """前向传播"""
        # 初始卷积
        out = self._conv2d(x, self.weights['conv1'], self.weights['conv1_bias'], stride=1)
        out = self._batch_norm(out,self.weights['batchnorm1'])
        out = self._relu(out)
        # 最大池化（简化实现）
        # out = out[:, :, ::2, ::2]  # 简单下采样代替maxpool

        # 残差块1
        out = self._residual_block(out, 'res1', stride=2)
        # 残差块2
        out = self._residual_block(out, 'res2', stride=1)
        # 残差块3
        out = self._residual_block(out, 'res3', stride=2)
        # 残差块4
        out = self._residual_block(out, 'res4', stride=1)

        # 全局平均池化
        out = self._global_avg_pool(out)

        # 全连接层（需要reshape）
        out = np.dot(out.reshape(out.shape[0], -1), self.weights['fc']) + self.weights['fc_bias']

        out = self.softmax(out)

        self.pred = out

    def loss(self):
        """计算交叉熵损失"""
        loss = - np.log(self.pred[range(len(self.pred)), self.label])
        return loss

    def accuracy(self):  # @save
        """计算预测正确的数量"""
        if len(self.pred.shape) > 1 and self.pred.shape[1] > 1:
            self.pred = self.pred.argmax(axis=1)
        cmp = self.pred.type(self.label.dtype) == self.label
        return float(cmp.type(self.label.dtype).sum())


    def backward(self):

        batch_size = self.label.shape[0]
        grad = np.zeros_like(self.pred)
        grad[range(batch_size), self.label] = 1
        grad -= self.pred
        grad /= batch_size


        return
