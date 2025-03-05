import torch
from pyemd import emd_with_flow
import numpy as np
import logging

logger = logging.getLogger()

def get_new_layer_weight(trans_matrix, distance_matrix, stu_layer_num, tea_layer_num, T, type_update='att'):
    if type_update == 'att':
        global att_student_weight, att_teacher_weight
        student_layer_weight = np.copy(att_student_weight)
        teacher_layer_weight = np.copy(att_teacher_weight)
    else:
        global rep_student_weight, rep_teacher_weight
        student_layer_weight = np.copy(rep_student_weight)
        teacher_layer_weight = np.copy(rep_teacher_weight)

    distance_matrix = distance_matrix.detach().cpu().numpy().astype('float64')
    trans_weight = np.sum(trans_matrix * distance_matrix, -1)
    # logger.info('student_trans_weight:{}'.format(trans_weight))
    # new_student_weight = torch.zeros(stu_layer_num).cuda()
    for i in range(stu_layer_num):
        student_layer_weight[i] = trans_weight[i] / student_layer_weight[i]
    weight_sum = np.sum(student_layer_weight)
    for i in range(stu_layer_num):
        if student_layer_weight[i] != 0:
            student_layer_weight[i] = weight_sum / student_layer_weight[i]

    trans_weight = np.sum(np.transpose(trans_matrix) * distance_matrix, -1)
    for j in range(tea_layer_num):
        teacher_layer_weight[j] = trans_weight[j + stu_layer_num] / teacher_layer_weight[j]
    weight_sum = np.sum(teacher_layer_weight)
    for i in range(tea_layer_num):
        if teacher_layer_weight[i] != 0:
            teacher_layer_weight[i] = weight_sum / teacher_layer_weight[i]

    student_layer_weight = student_layer_weight / np.sum(student_layer_weight)
    teacher_layer_weight = teacher_layer_weight / np.sum(teacher_layer_weight)

    if type_update == 'att':
        att_student_weight = student_layer_weight
        att_teacher_weight = teacher_layer_weight
    else:
        rep_student_weight = student_layer_weight
        rep_teacher_weight = teacher_layer_weight

def transformer_loss(student_atts, teacher_atts, student_reps, teacher_reps,
                     device, loss_mse, args, global_step, T=1):

    # EMD蒸馏：在这段代码中，关键部分是通过 emd_with_flow 函数计算 EMD，并通过转换矩阵加权层级之间的损失。
    # 具体来说，emd_att_loss 和 emd_rep_loss 函数都通过计算每一层的MSE损失来衡量学生和教师之间的差异，
    # 随后通过 EMD 来进行加权，最终计算总损失。

    global att_student_weight, rep_student_weight, att_teacher_weight, rep_teacher_weight
    def embedding_rep_loss(student_reps, teacher_reps, student_layer_weight, teacher_layer_weight,
                     stu_layer_num, tea_layer_num, device, loss_mse):

        # 计算的是 表示（representation） 的损失
        # emd_with_flow：它通过最小化搬运成本来找到从学生和教师之间的表示或注意力的对应关系，并返回转换矩阵。
        # 这个转换矩阵表征了从学生到教师的层级或注意力之间的“搬运流”。
        student_layer_weight = np.concatenate((student_layer_weight, np.zeros(tea_layer_num)))
        teacher_layer_weight = np.concatenate((np.zeros(stu_layer_num), teacher_layer_weight))
        totol_num = stu_layer_num + tea_layer_num
        distance_matrix = torch.zeros([totol_num, totol_num]).cuda()
        for i in range(stu_layer_num):
            student_rep = student_reps[i]
            for j in range(tea_layer_num):
                teacher_rep = teacher_reps[j]
                tmp_loss = loss_mse(student_rep, teacher_rep)
                distance_matrix[i][j + stu_layer_num] = distance_matrix[j + stu_layer_num][i] = tmp_loss

        _, trans_matrix = emd_with_flow(student_layer_weight, teacher_layer_weight,
                                        distance_matrix.detach().cpu().numpy().astype('float64'))
        print(np.array(trans_matrix).shape)
        print(np.array(distance_matrix).shape)
        rep_loss = torch.sum(torch.tensor(trans_matrix).cuda() * distance_matrix)
        return rep_loss, trans_matrix, distance_matrix

    def emd_rep_loss(student_reps, teacher_reps, student_layer_weight, teacher_layer_weight,
                     stu_layer_num, tea_layer_num, device, loss_mse):
        student_layer_weight = np.concatenate((student_layer_weight, np.zeros(tea_layer_num)))
        teacher_layer_weight = np.concatenate((np.zeros(stu_layer_num), teacher_layer_weight))
        totol_num = stu_layer_num + tea_layer_num
        distance_matrix = torch.zeros([totol_num, totol_num]).cuda()
        for i in range(stu_layer_num):
            student_rep = student_reps[i+1]
            for j in range(tea_layer_num):
                teacher_rep = teacher_reps[j + 1]
                tmp_loss = loss_mse(student_rep, teacher_rep)
                # tmp_loss = torch.nn.functional.normalize(tmp_loss, p=2, dim=2)
                distance_matrix[i][j + stu_layer_num] = distance_matrix[j + stu_layer_num][i] = tmp_loss

        _, trans_matrix = emd_with_flow(student_layer_weight, teacher_layer_weight,
                                        distance_matrix.detach().cpu().numpy().astype('float64'))
        # trans_matrix = trans_matrix
        rep_loss = torch.sum(torch.tensor(trans_matrix).cuda() * distance_matrix)
        return rep_loss, trans_matrix, distance_matrix

    def emd_att_loss(student_atts, teacher_atts, student_layer_weight, teacher_layer_weight,
                     stu_layer_num, tea_layer_num, device, loss_mse):

        #计算的是 注意力（attention） 的损失

        student_layer_weight = np.concatenate((student_layer_weight, np.zeros(tea_layer_num)))
        teacher_layer_weight = np.concatenate((np.zeros(stu_layer_num), teacher_layer_weight))
        totol_num = stu_layer_num + tea_layer_num
        distance_matrix = torch.zeros([totol_num, totol_num]).cuda()
        for i in range(stu_layer_num):
            student_att = student_atts[i]
            for j in range(tea_layer_num):
                teacher_att = teacher_atts[j]
                student_att = torch.where(student_att <= -1e2, torch.zeros_like(student_att).to(device),
                                          student_att)
                teacher_att = torch.where(teacher_att <= -1e2, torch.zeros_like(teacher_att).to(device),
                                          teacher_att)

                tmp_loss = loss_mse(student_att, teacher_att)
                distance_matrix[i][j + stu_layer_num] = distance_matrix[j + stu_layer_num][i] = tmp_loss
        _, trans_matrix = emd_with_flow(student_layer_weight, teacher_layer_weight,
                                        distance_matrix.detach().cpu().numpy().astype('float64'))
        att_loss = torch.sum(torch.tensor(trans_matrix).cuda() * distance_matrix)
        return att_loss, trans_matrix, distance_matrix

    stu_layer_num = len(student_atts)
    tea_layer_num = len(teacher_atts)
    if args.use_att:
        att_loss, att_trans_matrix, att_distance_matrix = \
            emd_att_loss(student_atts, teacher_atts, att_student_weight, att_teacher_weight,
                         stu_layer_num, tea_layer_num, device, loss_mse)
        if args.update_weight:
            get_new_layer_weight(att_trans_matrix, att_distance_matrix, stu_layer_num, tea_layer_num, T=T)
        att_loss = att_loss.to(device)
    else:
        att_loss = torch.tensor(0)
    if args.use_rep:
        rep_loss, rep_trans_matrix, rep_distance_matrix = \
            emd_rep_loss(student_reps, teacher_reps, rep_student_weight, rep_teacher_weight,
                            stu_layer_num, tea_layer_num, device, loss_mse)

        if args.update_weight:
            get_new_layer_weight(rep_trans_matrix, rep_distance_matrix, stu_layer_num, tea_layer_num, T=T, type_update='xx')
        rep_loss = rep_loss.to(device)
    else:
        rep_loss = torch.tensor(0)


    if not args.seperate:
        student_weight = np.mean(np.stack([att_student_weight, rep_student_weight]), 0)
        teacher_weight = np.mean(np.stack([att_teacher_weight, rep_teacher_weight]), 0)
        if global_step % args.eval_step == 0:
            logger.info('all_student_weight:{}'.format(student_weight))
            logger.info('all_teacher_weight:{}'.format(teacher_weight))
        att_student_weight = student_weight
        att_teacher_weight = teacher_weight
        rep_student_weight = student_weight
        rep_teacher_weight = teacher_weight
    else:
        if global_step % args.eval_step == 0:
            logger.info('att_student_weight:{}'.format(att_student_weight))
            logger.info('att_teacher_weight:{}'.format(att_teacher_weight))
            logger.info('rep_student_weight:{}'.format(rep_student_weight))
            logger.info('rep_teacher_weight:{}'.format(rep_teacher_weight))

#       att_student_weight = att_student_weight / np.sum(att_student_weight)
#       att_teacher_weight = att_teacher_weight / np.sum(att_teacher_weight)

#       rep_student_weight = rep_student_weight / np.sum(rep_student_weight)
#       rep_teacher_weight = rep_teacher_weight / np.sum(rep_student_weight)
    return att_loss, rep_loss

def pkd_loss(student_atts, teacher_atts, student_reps, teacher_reps, device='cuda'):
    teacher_atts = [teacher_atts[i] for i in [1, 3, 5, 7, 9]]
    att_tmp_loss, rep_tmp_loss = [], []
    for student_att, teacher_att in zip(student_atts, teacher_atts):
        student_att = torch.where(student_att <= -1e2, torch.zeros_like(student_att).to(device),
                                    student_att)
        teacher_att = torch.where(teacher_att <= -1e2, torch.zeros_like(teacher_att).to(device),
                                    teacher_att)

        att_tmp_loss.append(torch.nn.functional.mse_loss(student_att, teacher_att))
    att_loss = sum(att_tmp_loss)
    new_teacher_reps = [teacher_reps[i] for i in [2, 4, 6, 8, 10]]
    new_student_reps = student_reps[1:-1]
    for student_rep, teacher_rep in zip(new_student_reps, new_teacher_reps):
        rep_tmp_loss.append(torch.nn.functional.mse_loss(student_rep, teacher_rep))
    rep_loss = sum(rep_tmp_loss)

    return att_loss, rep_loss