import torch.nn as nn

from modules.transformation import TPS_SpatialTransformerNetwork
from modules.feature_extraction import VGG_FeatureExtractor, RCNN_FeatureExtractor, ResNet_FeatureExtractor
from modules.sequence_modeling import BidirectionalLSTM
from modules.prediction import Attention


class Model(nn.Module):

    def __init__(self, opt, SelfSL_layer=False):
        super(Model, self).__init__()
        self.opt = opt
        self.stages = {'Trans': opt.Transformation, 'Feat': opt.FeatureExtraction,
                       'Seq': opt.SequenceModeling, 'Pred': opt.Prediction}

        """ Transformation """
        if opt.Transformation == 'TPS':
            self.Transformation = TPS_SpatialTransformerNetwork(
                F=opt.num_fiducial, I_size=(opt.imgH, opt.imgW), I_r_size=(opt.imgH, opt.imgW), I_channel_num=opt.input_channel)
        else:
            print('No Transformation module specified')

        """ FeatureExtraction """
        if opt.FeatureExtraction == 'VGG':
            self.FeatureExtraction = VGG_FeatureExtractor(opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'RCNN':
            self.FeatureExtraction = RCNN_FeatureExtractor(opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'ResNet':
            self.FeatureExtraction = ResNet_FeatureExtractor(opt.input_channel, opt.output_channel)
        else:
            raise Exception('No FeatureExtraction module specified')
        self.FeatureExtraction_output = opt.output_channel
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1

        if not SelfSL_layer:  # for STR
            """ Sequence modeling"""
            if opt.SequenceModeling == 'BiLSTM':
                self.SequenceModeling = nn.Sequential(
                    BidirectionalLSTM(self.FeatureExtraction_output, opt.hidden_size, opt.hidden_size),
                    BidirectionalLSTM(opt.hidden_size, opt.hidden_size, opt.hidden_size))
                self.SequenceModeling_output = opt.hidden_size
            else:
                print('No SequenceModeling module specified')
                self.SequenceModeling_output = self.FeatureExtraction_output

        if not SelfSL_layer:  # for STR.
            """ Prediction """
            if opt.Prediction == 'CTC':
                self.Prediction = nn.Linear(self.SequenceModeling_output, opt.num_class)
            elif opt.Prediction == 'Attn':
                self.Prediction = Attention(self.SequenceModeling_output, opt.hidden_size, opt.num_class)
            else:
                raise Exception('Prediction is neither CTC or Attn')

        else:
            """ for self-supervised learning (SelfSL) """
            self.AdaptiveAvgPool_2 = nn.AdaptiveAvgPool2d((None, 1))  # make width -> 1
            if SelfSL_layer == 'CNN':
                self.SelfSL_FFN_input = self.FeatureExtraction_output

            if 'RotNet' in self.opt.self:
                self.SelfSL = nn.Linear(self.SelfSL_FFN_input, 4)  # 4 = [0, 90, 180, 270] degrees
            elif 'MoCo' in self.opt.self:
                self.SelfSL = nn.Linear(self.SelfSL_FFN_input, 128)  # 128 is used for MoCo paper.

    def forward(self, image, text=None, is_train=True, SelfSL_layer=False):
        """ Transformation stage """
        if not self.stages['Trans'] == 'None':
            image = self.Transformation(image)

        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(image)
        visual_feature = visual_feature.permute(0, 3, 1, 2)    # [b, c, h, w] -> [b, w, c, h]
        visual_feature = self.AdaptiveAvgPool(visual_feature)  # [b, w, c, h] -> [b, w, c, 1]
        visual_feature = visual_feature.squeeze(3)             # [b, w, c, 1] -> [b, w, c]

        """ for self supervised learning on Feature extractor (CNN part) """
        if SelfSL_layer == 'CNN':
            visual_feature = visual_feature.permute(0, 2, 1)         # [b, w, c] -> [b, c, w]
            visual_feature = self.AdaptiveAvgPool_2(visual_feature)  # [b, c, w] -> [b, c, 1]
            visual_feature = visual_feature.squeeze(2)               # [b, c, 1] -> [b, c]
            prediction_SelfSL = self.SelfSL(visual_feature)          # [b, c] -> [b, SelfSL_class]
            return prediction_SelfSL

        """ Sequence modeling stage """
        if self.stages['Seq'] == 'BiLSTM':
            contextual_feature = self.SequenceModeling(visual_feature)  # [b, num_steps, opt.hidden_size]
        else:
            contextual_feature = visual_feature  # for convenience. this is NOT contextually modeled by BiLSTM

        """ Prediction stage """
        if self.stages['Pred'] == 'CTC':
            prediction = self.Prediction(contextual_feature.contiguous())
        else:
            prediction = self.Prediction(contextual_feature.contiguous(), text, is_train,
                                         batch_max_length=self.opt.batch_max_length)

        return prediction  # [b, num_steps, opt.num_class]
