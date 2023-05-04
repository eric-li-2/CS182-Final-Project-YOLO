import torch
import torch.nn as nn

architecture_config = [
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))


class Yolov1(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super(Yolov1, self).__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.architecture)
        self.fcs = self._create_fcs(**kwargs)

    def forward(self, x):
        x = self.darknet(x)
        return self.fcs(torch.flatten(x, start_dim=1))

    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == tuple:
                layers += [
                    CNNBlock(
                        in_channels, x[1], kernel_size=x[0], stride=x[2], padding=x[3],
                    )
                ]
                in_channels = x[1]

            elif type(x) == str:
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

            elif type(x) == list:
                conv1 = x[0]
                conv2 = x[1]
                num_repeats = x[2]

                for _ in range(num_repeats):
                    layers += [
                        CNNBlock(
                            in_channels,
                            conv1[1],
                            kernel_size=conv1[0],
                            stride=conv1[2],
                            padding=conv1[3],
                        )
                    ]
                    layers += [
                        CNNBlock(
                            conv1[1],
                            conv2[1],
                            kernel_size=conv2[0],
                            stride=conv2[2],
                            padding=conv2[3],
                        )
                    ]
                    in_channels = conv2[1]

        return nn.Sequential(*layers)

    def _create_fcs(self, S, B, C):
        # In original paper this should be
        # nn.Linear(1024*S*S, 4096),
        # nn.LeakyReLU(0.1),
        # nn.Linear(4096, S*S*(B*5+C))

        ### YOUR CODE HERE
        # Return a module that carries out a fully connected layer
        # 1. Flatten the output from the previous layer
        # 2. Add a fully connected layer with 496 output features
        # 3. Add a LeakyReLU activation with slope 0.1
        # 4. Add a fully connected layer with the right number of output features. 
        # HINT: look at figure 2 of the paper
        # return NotImplementedError()        
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 496),
            # nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(496, S * S * (C + B * 5)),
        )
        ### END CODE HERE


class BaselineClassifier(nn.Module):
    """
    Baseline model that slides classifier along image
    """
    def __init__(self, in_channels=3, **kwargs):
        super(BaselineClassifier, self).__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.architecture)
        self.fcs = self._create_fcs(**kwargs)

    def forward(self, x):
        x = self.darknet(x)
        return self.fcs(torch.flatten(x, start_dim=1))

    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == tuple:
                layers += [
                    CNNBlock(
                        in_channels, x[1], kernel_size=x[0], stride=x[2], padding=x[3],
                    )
                ]
                in_channels = x[1]

            elif type(x) == str:
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

            elif type(x) == list:
                conv1 = x[0]
                conv2 = x[1]
                num_repeats = x[2]

                for _ in range(num_repeats):
                    layers += [
                        CNNBlock(
                            in_channels,
                            conv1[1],
                            kernel_size=conv1[0],
                            stride=conv1[2],
                            padding=conv1[3],
                        )
                    ]
                    layers += [
                        CNNBlock(
                            conv1[1],
                            conv2[1],
                            kernel_size=conv2[0],
                            stride=conv2[2],
                            padding=conv2[3],
                        )
                    ]
                    in_channels = conv2[1]

        return nn.Sequential(*layers)

    def _create_fcs(self, num_classes=3):
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(16384, 496),
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(496, num_classes),
        )

class BaselineObjectDetector(nn.Module):
    def __init__(self, clf, clf_dim=64*4):
        """
        Baseline object detector using classifier on sliding window across image

        Args:
            clf: Pytorch classifier
            clf_dim: Input image size of classifier (assume square image)
        """
        super(BaselineObjectDetector, self).__init__()
        self.clf = clf
        self.clf_dim = clf_dim
        self.stride = clf_dim // 4
        self.unfold = nn.Unfold(kernel_size=(clf_dim, clf_dim), stride=self.stride)

    def forward(self, x):
        batch_size, in_channels, in_height, in_width = x.shape

        out_height = int((in_height - self.clf_dim) / self.stride + 1)
        out_width = int((in_width - self.clf_dim) / self.stride + 1)

        # Get sliding window patches (similar to im2col from convolution)
        window_patches = self.unfold(x) # [bsz, in_channels * clf_dim * clf_dim, out_height * out_width]

        # Arrage patches for classifier
        input_patches = window_patches.permute(0, 2, 1).reshape(
            batch_size * out_height * out_width, in_channels, self.clf_dim, self.clf_dim
        ) # [bsz * out_height * out_width, in_channels, clf_dim, clf_dim]

        ### YOUR CODE HERE
        # 1. Run classifier on all input patches to predict logits for 3 classes (square, circle, or nothing) per patch
        # The output of the classifier should have shape [bsz * out_height * out_width, num_classes=3]
        # 2. Reshape the logits to be [bsz, out_height, out_width, num_classes=3]. 
        # This is to bring back the grid structure
        # 3. Run softmax on the last dimension (num_classes) to get class probabilities per patch
        # HINT: Use torch reshape and softmax
        clf_out = self.clf(input_patches) # [bsz * out_height * out_width, num_classes]
        clf_out = clf_out.reshape(batch_size, out_height, out_width, -1) # [bsz, out_height, out_width, num_classes]
        clf_out = torch.softmax(clf_out, axis=-1) # [bsz, out_height, out_width, num_classes]
        ### END CODE HERE

        return clf_out
