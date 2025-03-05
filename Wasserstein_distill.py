import torch
import numpy as np
from pyemd import emd_with_flow# 从pyemd库导入emd_with_flow函数，用于计算地球移动距离（EMD）及流量矩阵
import logging

logger = logging.getLogger()


# 定义函数emd_distance，用于计算学生和教师之间的EMD距离
def emd_distance(student_reps, teacher_reps, student_layer_weight, teacher_layer_weight,
                 stu_layer_num, tea_layer_num, device, loss_mse):
    # 将学生层权重和教师层权重进行拼接，保证权重数组长度一致
    # 学生层权重后面拼接与教师层数相同的零
    student_layer_weight = np.concatenate((student_layer_weight, np.zeros(tea_layer_num)))
    # 教师层权重前面拼接与学生层数相同的零
    teacher_layer_weight = np.concatenate((np.zeros(stu_layer_num), teacher_layer_weight))
    # 计算总层数（学生层数+教师层数）
    total_num = stu_layer_num + tea_layer_num
    # 初始化一个大小为(total_num, total_num)的零矩阵，用于存储各层之间的距离，并转移到指定设备上
    distance_matrix = torch.zeros([total_num, total_num]).to(device)

    # 遍历学生的每一层表示
    for i in range(stu_layer_num):
        # 获取学生第i层的表示
        student_rep = student_reps[i]
        # 遍历教师的每一层表示
        for j in range(tea_layer_num):
            # 获取教师第j层的表示
            teacher_rep = teacher_reps[j]
            # 计算学生层和教师层之间的均方误差（MSE）损失，作为距离
            tmp_loss = loss_mse(student_rep, teacher_rep)
            # 将计算得到的距离赋值到对称的位置上
            distance_matrix[i][j + stu_layer_num] = distance_matrix[j + stu_layer_num][i] = tmp_loss

    # 计算EMD流量，返回流量矩阵trans_matrix
    # 注意：需要将distance_matrix转移到CPU并转换为float64类型进行计算
    _, trans_matrix = emd_with_flow(student_layer_weight, teacher_layer_weight,
                                    distance_matrix.detach().cpu().numpy().astype('float64'))

    # 根据流量矩阵和距离矩阵计算EMD损失：各项相乘后求和
    emd_loss = torch.sum(torch.tensor(trans_matrix).to(device) * distance_matrix)

    # 返回EMD损失、流量矩阵和距离矩阵
    return emd_loss, trans_matrix, distance_matrix


# 定义wasserstein_loss函数，用于计算基于注意力和表示的Wasserstein距离损失
def wasserstein_loss(student_atts, teacher_atts, student_reps, teacher_reps,
                     device, loss_mse, args, global_step, T=1):
    # 声明全局变量，用于存储注意力和表示的学生/教师层权重
    global att_student_weight, rep_student_weight, att_teacher_weight, rep_teacher_weight

    # 获取学生和教师注意力层的数量
    stu_layer_num = len(student_atts)
    tea_layer_num = len(teacher_atts)

    # 如果使用注意力信息（args.use_att为True）
    if args.use_att:
        # 计算注意力部分的EMD损失及对应的流量和距离矩阵
        att_loss, att_trans_matrix, att_distance_matrix = \
            emd_distance(student_atts, teacher_atts, att_student_weight, att_teacher_weight,
                         stu_layer_num, tea_layer_num, device, loss_mse)

        # 如果需要更新权重（args.update_weight为True），则调用get_new_layer_weight函数更新注意力权重
        if args.update_weight:
            get_new_layer_weight(att_trans_matrix, att_distance_matrix, stu_layer_num, tea_layer_num, T=T)
        # 将计算的注意力损失转移到指定设备上
        att_loss = att_loss.to(device)
    else:
        # 如果不使用注意力信息，则注意力损失设为0
        att_loss = torch.tensor(0).to(device)

    # 如果使用表示信息（args.use_rep为True）
    if args.use_rep:
        # 计算表示部分的EMD损失及对应的流量和距离矩阵
        rep_loss, rep_trans_matrix, rep_distance_matrix = \
            emd_distance(student_reps, teacher_reps, rep_student_weight, rep_teacher_weight,
                         stu_layer_num, tea_layer_num, device, loss_mse)
        # 如果需要更新权重（args.update_weight为True），则调用get_new_layer_weight函数更新表示权重
        if args.update_weight:
            get_new_layer_weight(rep_trans_matrix, rep_distance_matrix, stu_layer_num, tea_layer_num, T=T)
        # 将计算的表示损失转移到指定设备上
        rep_loss = rep_loss.to(device)
    else:
        # 如果不使用表示信息，则表示损失设为0
        rep_loss = torch.tensor(0).to(device)

    # 如果不分开更新注意力和表示的权重（args.seperate为False）
    if not args.seperate:
        # 计算学生权重为注意力权重和表示权重的均值
        student_weight = np.mean(np.stack([att_student_weight, rep_student_weight]), axis=0)
        # 计算教师权重为注意力权重和表示权重的均值
        teacher_weight = np.mean(np.stack([att_teacher_weight, rep_teacher_weight]), axis=0)
        # 每隔一定步数记录日志，输出合并后的权重信息
        if global_step % args.eval_step == 0:
            logger.info(f'all_student_weight: {student_weight}')
            logger.info(f'all_teacher_weight: {teacher_weight}')
        # 更新全局注意力和表示的学生权重
        att_student_weight, rep_student_weight = student_weight, student_weight
        # 更新全局注意力和表示的教师权重
        att_teacher_weight, rep_teacher_weight = teacher_weight, teacher_weight
    else:
        # 如果分开更新注意力和表示的权重，每隔一定步数分别记录各自的权重信息
        if global_step % args.eval_step == 0:
            logger.info(f'att_student_weight: {att_student_weight}')
            logger.info(f'att_teacher_weight: {att_teacher_weight}')
            logger.info(f'rep_student_weight: {rep_student_weight}')
            logger.info(f'rep_teacher_weight: {rep_teacher_weight}')
    # 返回注意力部分和表示部分的损失
    return att_loss, rep_loss
