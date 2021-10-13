import torch
import torch.nn as nn
import torch.nn.functional as F

from model import Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PseudoLabelLoss(nn.Module):
    def __init__(self, opt, converter, criterion):
        super(PseudoLabelLoss, self).__init__()

        self.opt = opt
        self.converter = converter
        self.criterion = criterion

        self.PseudoLabel_prediction_model = Model(opt)
        self.PseudoLabel_prediction_model = torch.nn.DataParallel(
            self.PseudoLabel_prediction_model
        ).to(device)
        print(
            f"### loading pretrained model for PseudoLabel from {opt.model_for_PseudoLabel}"
        )
        self.PseudoLabel_prediction_model.load_state_dict(
            torch.load(opt.model_for_PseudoLabel)
        )
        self.PseudoLabel_prediction_model.eval()

    def forward(self, image_unlabel, model):

        with torch.no_grad():
            if "CTC" in self.opt.Prediction:
                PseudoLabel_pred = self.PseudoLabel_prediction_model(image_unlabel)
            else:
                idx_for_pred = (
                    torch.LongTensor(image_unlabel.size(0))
                    .fill_(self.opt.sos_token_index)
                    .to(device)
                )
                PseudoLabel_pred = self.PseudoLabel_prediction_model(
                    image_unlabel, text=idx_for_pred, is_train=False
                )

        _, PseudoLabel_index = PseudoLabel_pred.max(2)
        length_for_decode = torch.IntTensor(
            [PseudoLabel_pred.size(1)] * PseudoLabel_pred.size(0)
        )
        PseudoLabel_tmp = self.converter.decode(PseudoLabel_index, length_for_decode)
        PseudoLabel = []
        image_unlabel_unbind = torch.unbind(image_unlabel, dim=0)
        image_unlabel_filtered = []
        for image_ul, pseudo_label in zip(image_unlabel_unbind, PseudoLabel_tmp):
            # filtering unlabeled images whose prediction containing [PAD], [UNK], or [SOS] token.
            if (
                "[PAD]" in pseudo_label
                or "[UNK]" in pseudo_label
                or "[SOS]" in pseudo_label
            ):
                continue
            else:
                if "Attn" in self.opt.Prediction:
                    index_EOS = pseudo_label.find("[EOS]")
                    pseudo_label = pseudo_label[:index_EOS]

                PseudoLabel.append(pseudo_label)
                image_unlabel_filtered.append(image_ul)

        image_unlabel_filtered = torch.stack(image_unlabel_filtered, dim=0)
        Pseudo_index, Pseudo_length = self.converter.encode(
            PseudoLabel, batch_max_length=self.opt.batch_max_length
        )

        if "CTC" in self.opt.Prediction:
            preds_PL = model(image_unlabel_filtered)
            preds_PL_size = torch.IntTensor([preds_PL.size(1)] * preds_PL.size(0))
            preds_PL_log_softmax = preds_PL.log_softmax(2).permute(1, 0, 2)
            loss_SemiSL = self.criterion(
                preds_PL_log_softmax, Pseudo_index, preds_PL_size, Pseudo_length
            )
        else:
            preds_PL = model(
                image_unlabel_filtered, Pseudo_index[:, :-1]
            )  # align with Attention.forward
            target = Pseudo_index[:, 1:]  # without [SOS] Symbol
            loss_SemiSL = self.criterion(
                preds_PL.view(-1, preds_PL.shape[-1]), target.contiguous().view(-1)
            )

        return loss_SemiSL


class MeanTeacherLoss(nn.Module):
    """
    check authors code from https://github.com/CuriousAI/mean-teacher/tree/master/pytorch
    """

    def __init__(self, opt, student_for_init_teacher):
        super(MeanTeacherLoss, self).__init__()

        self.opt = opt
        self.teacher = Model(opt)  # create the ema model
        self.teacher = torch.nn.DataParallel(self.teacher).to(device)
        self.teacher.train()
        self.criterion = torch.nn.MSELoss().to(device)
        if opt.Prediction == "Attn":
            self.text_for_pred = (
                torch.LongTensor(opt.batch_size).fill_(opt.sos_token_index).to(device)
            )

        # copy student and init
        for param_t, param_s in zip(
            self.teacher.parameters(), student_for_init_teacher.parameters()
        ):
            param_t.data.copy_(param_s.data)  # initialize
            param_t.requires_grad = False  # not update by gradient

    def _update_ema_variables(self, student, iteration, alpha=0.999):
        # Use the true average until the exponential average is more correct
        alpha = min(1 - 1 / (iteration + 1), alpha)
        for param_t, param_s in zip(self.teacher.parameters(), student.parameters()):
            param_t.data = param_t.data * alpha + param_s.data * (1.0 - alpha)

    def forward(self, student_input, student_logit, student, teacher_input, iteration):

        image_label_size = student_logit.size(0)
        image_unlabel = student_input[image_label_size:]
        if "CTC" in self.opt.Prediction:
            student_logit_unlabel = student(image_unlabel)
        else:
            student_logit_unlabel = student(
                image_unlabel, text=self.text_for_pred, is_train=False
            )
        student_logit = torch.cat([student_logit, student_logit_unlabel], dim=0)

        with torch.no_grad():
            self._update_ema_variables(student, iteration, self.opt.MT_alpha)
            if "CTC" in self.opt.Prediction:
                teacher_logit = self.teacher(teacher_input)
            else:
                teacher_logit = self.teacher(
                    teacher_input, text=self.text_for_pred, is_train=False
                )

        softmax_student_logit = F.softmax(student_logit, dim=2)
        softmax_teacher_logit = F.softmax(teacher_logit, dim=2)

        loss_SemiSL = self.opt.MT_C * self.criterion(
            softmax_student_logit, softmax_teacher_logit
        )

        return loss_SemiSL
