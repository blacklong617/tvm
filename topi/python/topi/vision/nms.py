# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=import-error, invalid-name, no-member, too-many-locals, too-many-arguments, undefined-variable, too-many-nested-blocks, too-many-branches, too-many-statements, too-many-function-args
"""Non-maximum suppression operator"""
import tvm

from tvm import hybrid
from ..sort import argsort

@hybrid.script
def hybrid_rearrange_out(data):
    """Hybrid routine to rearrange nms output to
    move all valid entries to top.

    Parameters
    ----------
    data : tvm.Tensor or numpy NDArray
        NMS output. 3-D tensor with shape
        [batch_size, num_anchors, 6].

    Returns
    -------
    output : tvm.Tensor or numpy NDArray
        Transformed NMS output. 3-D tensor with shape
        [batch_size, num_anchors, 6].
    """
    batch_size = data.shape[0]
    num_anchors = data.shape[1]
    elem_length = data.shape[2]
    output = output_tensor((batch_size,
                            num_anchors,
                            elem_length),
                           data.dtype)

    for i in parallel(batch_size):
        valid_idx = 0
        for j in range(num_anchors):
            if data[i, j, 0] >= 0:
                for k in range(elem_length):
                    output[i, valid_idx, k] = data[i, j, k]
                valid_idx += 1
            if j >= valid_idx:
                for k in range(elem_length):
                    output[i, j, k] = -1.0
    return output


@hybrid.script
def hybrid_get_valid_counts(data, score_threshold, id_index, score_index):
    """Hybrid routine to get valid count of bounding boxes
    given a score threshold. Also moves valid boxes to the
    top of input data.

    Parameters
    ----------
    data : tvm.Tensor or numpy NDArray
        Input data. 3-D tensor with shape [batch_size, num_anchors, 6]
        or [batch_size, num_anchors, 5].

    score_threshold : tvm.const
        Lower limit of score for valid bounding boxes.

    id_index : tvm.const
        index of the class categories, -1 to disable.

    score_index: tvm.const
        Index of the scores/confidence of boxes.

    Returns
    -------
    out_tensor : tvm.Tensor or numpy NDArray
        Rearranged data tensor.

    valid_count : tvm.Tensor or numpy NDArray
        1-D tensor for valid number of boxes.
    """
    batch_size = data.shape[0]
    num_anchors = data.shape[1]
    box_data_length = data.shape[2]
    valid_count = output_tensor((batch_size,), "int32")
    out_tensor = output_tensor((batch_size,
                                num_anchors,
                                box_data_length),
                               data.dtype)
    for i in parallel(batch_size):
        valid_count[i] = 0
        for j in range(num_anchors):
            score = data[i, j, score_index]
            if score > score_threshold and \
                    (id_index < 0 or data[i, j, id_index] >= 0):
                for k in range(box_data_length):
                    out_tensor[i, valid_count[i], k] = data[i, j, k]
                valid_count[i] += 1
            if j >= valid_count[i]:
                for k in range(box_data_length):
                    out_tensor[i, j, k] = -1.0
    return valid_count, out_tensor

@tvm.target.generic_func
def get_valid_counts(data, score_threshold=0, id_index=0, score_index=1):
    """Get valid count of bounding boxes given a score threshold.
    Also moves valid boxes to the top of input data.

    Parameters
    ----------
    data : tvm.Tensor
        Input data. 3-D tensor with shape [batch_size, num_anchors, 6]
        or [batch_size, num_anchors, 5].

    score_threshold : optional, float
        Lower limit of score for valid bounding boxes.

    id_index : optional, int
        index of the class categories, -1 to disable.

    score_index: optional, int
        Index of the scores/confidence of boxes.

    Returns
    -------
    out_tensor : tvm.Tensor
        Rearranged data tensor.

    valid_count : tvm.Tensor
        1-D tensor for valid number of boxes.
    """
    score_threshold_const = tvm.const(score_threshold, "float32")
    id_index_const = tvm.const(id_index, "int32")
    score_index_const = tvm.const(score_index, "int32")
    return hybrid_get_valid_counts(data, score_threshold_const,
                                   id_index_const, score_index_const)


# add by lxz for onnx nms at 20190808
@hybrid.script
def hybrid_onnx_nms(boxes, scores, max_output_boxes_per_class, 
                iou_threshold, score_threshold):
    """
    boxes: [num_batches, spatial_dimension, 4]  coordinates 4  data [y1,x1,y2,x2]
    scores: [num_batches, num_classes, spatial_dimension]  
    spatial_dimension：每个batch中每个类别score的个数

    output: [num_selected_indices, 3]  coordinates 3  data [batch_index, class_index, box_index]

    """
    num_batch = boxes.shape[0]
    num_box = boxes.shape[1]
    num_class = scores.shape[1]
    output = output_tensor((num_batches * num_class * max_output_boxes_per_class, 3), "int32") 
    output_box = output_tensor((num_batches * num_class * max_output_boxes_per_class, 4), "int32")
    #  score_threshold_const = tvm.const(score_threshold, "float32")
    
    #  score 过滤 把比阈值低的score置为-1
    #  topk num_class num_box 
    if score_threshold >0:
        for i in range(num_batch):
            for j in range(num_class):
                for k in range(num_box):
                    #  过滤 低于阈值的score
                    if scores[i,j,k] < score_threshold:
                        scores[i,j,k] = -1
                    #  每个类最多输出的box为max_output_boxes_per_class个
                    #  则这里需要对每个类按照score由大到小进行排序后取前max_output_boxes_per_class个
    # sort_score 对每个类的box序列进行了编号，并按照score的大小进行了排序，sort_score 中存放排序好的序列号
    sort_score = argsort(scores, axis=2, is_ascend=False)  

                

    # iou 过滤
    
    for i in range(num_batch):
        mkeep = max_output_boxes_per_class
        # for j in range(num_class):
        if 0 < num_box < max_output_boxes_per_class:
            mkeep = num_box
      
        for j in range(mkeep):

            if iou_threshold > 0:

                box_a_idx = j
                for k in parallel(num_box):
                    a_y1 = boxes[batch_idx, box_a_idx, 0]  
                    a_x1 = boxes[batch_idx, box_a_idx, 1]  
                    a_y2 = boxes[batch_idx, box_a_idx, 2]  
                    a_x2 = boxes[batch_idx, box_a_idx, 3]  
                    box_b_idx = k

                    b_y1 = boxes[batch_idx, box_b_idx, 0]
                    b_x1 = boxes[batch_idx, box_b_idx, 1]
                    b_y2 = boxes[batch_idx, box_b_idx, 2]
                    b_x2 = boxes[batch_idx, box_b_idx, 3]

                    w = max(0.0, min(a_x2, b_x2) - max(a_x1, b_x1))  # max(0,min(a_x2,b_x2)-max(a_x1,b_x1))  求相交区域宽度
                    h = max(0.0, min(a_y2, b_y2) - max(a_y1, b_y1))  # max(0,min(a_y2,b_y2)-max(a_y1,b_y1))  求相交区域高度
                    area = h * w  # 相交区域面积
                    u = (a_x2 - a_x1) * (a_y2 - a_y1) + (b_x2 - b_x1) * (b_y2 - b_y1) - area  # 两个box相并后面积
                    iou = 0.0 if u <= 0.0 else area / u  # 重合面积占相并区域面积的百分比
                    if iou >= iou_threshold:  # 如果重合百分比大于等于阈值
                        output[i, k, score_index] = -1.0  # 将第k个box的score值置为-1
                        if id_index >= 0:  # 如果有class_id,把class_id置为-1
                            output[i, k, id_index] = -1.0
                        box_indices[i, k] = -1

    return output


@hybrid.script
def hybrid_nms(data, sorted_index, valid_count,
               max_output_size, iou_threshold, force_suppress,
               top_k, coord_start, id_index, score_index):
    """Hybrid routing for non-maximum suppression.

    Parameters
    ----------
    data: tvm.Tensor or numpy NDArray
        Bounding boxes with class and score. 3-D tensor with shape
        [batch_size, num_anchors, 6].

    sorted_index : tvm.Tensor or numpy NDArray
        Bounding box indexes sorted by score, with shape
        [batch_size, num_anchors].

    valid_count : tvm.Tensor or numpy NDArray
        1-D tensor for valid number of boxes.

    max_output_size : tvm.const
        Max number of output valid boxes for each instance.
        By default all valid boxes are returned.

    iou_threshold : tvm.const
        Overlapping(IoU) threshold to suppress object with smaller score.

    force_suppress : tvm.const
        Whether to suppress all detections regardless of class_id.

    top_k : tvm.const
        Keep maximum top k detections before nms, -1 for no limit.

    coord_start : tvm.const
        Start index of the consecutive 4 coordinates.

    id_index : tvm.const
        index of the class categories, -1 to disable.

    score_index: tvm.const
        Index of the scores/confidence of boxes.

    Returns
    -------
    output : tvm.Tensor
        3-D tensor with shape [batch_size, num_anchors, 6].

    box_indices: tvm.Tensorbatch_size
        2-D tensor with shape [batch_size, num_anchors].
    """
    batch_size = data.shape[0]  # 取出input data 中 batch_size 
    num_anchors = data.shape[1]  # 取出input data 中 anchor的数目
    box_data_length = data.shape[2]  # 取出input data 中 每个box的参数的个数
    box_indices = output_tensor((batch_size, num_anchors), "int32")  # 构造需要输出的tensor之一
    output = output_tensor((batch_size,
                            num_anchors,
                            box_data_length,),
                           data.dtype)  # 构造需要输出的tensor之一
    # 外循环为处理不同batch
    for i in range(batch_size):
        if iou_threshold > 0:  # 判断iou阈值是否大于零
            if valid_count[i] > 0:  # 判断当前batch的有效框是否大于零
                # Reorder output
                nkeep = valid_count[i]  # 取出当前batch的有效框个数
                if 0 < top_k < nkeep:  # 如果top_k的值小于当前batch有效框个数
                    nkeep = top_k  # 接下来的处理以top_k的值来处理
                for j in parallel(nkeep):  # 并行化处理每个batch的框
                    for k in range(box_data_length):  # 处理每个框的每个数据
                        output[i, j, k] = data[i, sorted_index[i, j], k]  # 取出boxes的data，按照score取前几个我们需要的个数
                    box_indices[i, j] = sorted_index[i, j]  # 记录box的索引信息[batch_index,box_index]
                if 0 < top_k < valid_count[i]:  # 这句应该是多余的，可以与上面的合并
                    for j in parallel(valid_count[i] - nkeep):  # 将output其他的位置（top_k以外）置为-1
                        for k in range(box_data_length):
                            output[i, j + nkeep, k] = -1.0
                        box_indices[i, j + nkeep] = -1  # 将box_indice输出的其他位置置为-1
            # Apply nms
            box_start_idx = coord_start  # 
            batch_idx = i
            for j in range(valid_count[i]):  # 处理每个有效的box
                # output[i, j, score_index] > 0  output中box的score大于零
                # id_index < 0  输入的id_index 小于零 即不需要class
                # output[i, j, id_index] >= 0  class_id 大于等于零
                if output[i, j, score_index] > 0 and (id_index < 0 or output[i, j, id_index] >= 0):
                    box_a_idx = j
                    # 处理box的 IOU
                    for k in parallel(valid_count[i]):
                        check_iou = 0
                        #  output[i, k, score_index] > 0 output中box的sco大于零
                        #  (id_index < 0 or output[i, k, id_index] >= 0) 输出不需要class或者class_id 大于等于零
                        if k > j and output[i, k, score_index] > 0 \
                                and (id_index < 0 or output[i, k, id_index] >= 0):
                            # 如果 force_suppress为 True
                            # 如果 不需要class或者两个为同一个class
                            # 即当force_suppress为True时或者id_index为-1时又或者两个要操作的box为同一个class时
                            if force_suppress:
                                check_iou = 1
                            elif id_index < 0 or output[i, j, id_index] == output[i, k, id_index]:
                                check_iou = 1
                        if check_iou > 0:
                            a_l = output[batch_idx, box_a_idx, box_start_idx]  #x1
                            a_t = output[batch_idx, box_a_idx, box_start_idx + 1]  #y1
                            a_r = output[batch_idx, box_a_idx, box_start_idx + 2]  #x2
                            a_b = output[batch_idx, box_a_idx, box_start_idx + 3]  #y2
                            box_b_idx = k
                            b_t = output[batch_idx, box_b_idx, box_start_idx + 1]
                            b_b = output[batch_idx, box_b_idx, box_start_idx + 3]
                            b_l = output[batch_idx, box_b_idx, box_start_idx]
                            b_r = output[batch_idx, box_b_idx, box_start_idx + 2]
                            w = max(0.0, min(a_r, b_r) - max(a_l, b_l))  # max(0,min(a_x2,b_x2)-max(a_x1,b_x1))  求相交区域宽度
                            h = max(0.0, min(a_b, b_b) - max(a_t, b_t))  # max(0,min(a_y2,b_y2)-max(a_y1,b_y1))  求相交区域高度
                            area = h * w  # 相交区域面积
                            u = (a_r - a_l) * (a_b - a_t) + (b_r - b_l) * (b_b - b_t) - area  # 两个box相并后面积
                            iou = 0.0 if u <= 0.0 else area / u  # 重合面积占相并区域面积的百分比
                            if iou >= iou_threshold:  # 如果重合百分比大于等于阈值
                                output[i, k, score_index] = -1.0  # 将第k个box的score值置为-1
                                if id_index >= 0:  # 如果有class_id,把class_id置为-1
                                    output[i, k, id_index] = -1.0
                                box_indices[i, k] = -1
        else:
            #iou 小于零，则不处理iou直接输出有效个box
            for j in parallel(valid_count[i]):
                for k in range(box_data_length):
                    output[i, j, k] = data[i, j, k]
                box_indices[i, j] = j
        # Set invalid entry to be -1
        for j in parallel(num_anchors - valid_count[i]):
            for k in range(box_data_length):
                output[i, j + valid_count[i], k] = -1.0
            box_indices[i, j + valid_count[i]] = -1
        # Only return max_output_size valid boxes
        num_valid_boxes = 0
        if max_output_size > 0:
            for j in parallel(valid_count[i]):
                if output[i, j, 0] >= 0:
                    if num_valid_boxes == max_output_size:
                        for k in range(box_data_length):
                            output[i, j, k] = -1.0
                        box_indices[i, j] = -1
                    else:
                        num_valid_boxes += 1
    return output, box_indices


@tvm.target.generic_func
def non_max_suppression(data, valid_count, max_output_size=-1,
                        iou_threshold=0.5, force_suppress=False, top_k=-1,
                        coord_start=2, score_index=1, id_index=0,
                        return_indices=True, invalid_to_bottom=False):
    """Non-maximum suppression operator for object detection.

    Parameters
    ----------
    data : tvm.Tensor
        3-D tensor with shape [batch_size, num_anchors, 6] or [batch_size, num_anchors, 5].

    valid_count : tvm.Tensor
        1-D tensor for valid number of boxes.

    max_output_size : optional, int
        Max number of output valid boxes for each instance.
        By default all valid boxes are returned.

    iou_threshold : optional, float
        Non-maximum suppression threshold.

    force_suppress : optional, boolean
        Whether to suppress all detections regardless of class_id.

    top_k : optional, int
        Keep maximum top k detections before nms, -1 for no limit.

    coord_start : required, int
        Start index of the consecutive 4 coordinates.

    score_index: optional, int
        Index of the scores/confidence of boxes.

    id_index : optional, int
        index of the class categories, -1 to disable.

    return_indices : optional, boolean
        Whether to return box indices in input data.

    invalid_to_bottom : optional, boolean
        Whether to mbox_leftid bounding boxes to the top.

    Returns
    -------
    out : tvm.Tensorbox_left
        3-D tensor with shape [batch_size, num_anchors, 6].

    Example
    --------
    .. code-block:: python

        # An example to use non_max_suppression
        dshape = (1, 5, 6)
        data = tvm.placeholder(dshape, name="data")
        valid_count = tvm.placeholder((dshape[0],), dtype="int32", name="valid_count")
        iou_threshold = 0.7
        force_suppress = True
        top_k = -1
        out = non_max_suppression(data, valid_count, iou_threshold=iou_threshold,
                                  force_suppress=force_suppress, top_k=top_k)
        np_data = np.random.uniform(dshape)
        np_valid_count = np.array([4])
        s = topi.generic.schedule_nms(out)
        f = tvm.build(s, [data, valid_count, out], "llvm")
        ctx = tvm.cpu()
        tvm_data = tvm.nd.array(np_data, ctx)
        tvm_valid_count = tvm.nd.array(np_valid_count, ctx)
        tvm_out = tvm.nd.array(np.zeros(dshape, dtype=data.dtype), ctx)
        f(tvm_data, tvm_valid_count, tvm_out)
    """
    batch_size = data.shape[0]
    num_anchors = data.shape[1]
    score_axis = score_index
    score_shape = (batch_size, num_anchors)
    score_tensor = tvm.compute(score_shape, lambda i, j: data[i, j, score_axis])
    sort_tensor = argsort(score_tensor, valid_count=valid_count, axis=1, is_ascend=False)
    out, box_indices = hybrid_nms(data, sort_tensor, valid_count,
                                  tvm.const(max_output_size, dtype="int32"),
                                  tvm.const(iou_threshold, dtype="float32"),
                                  tvm.const(force_suppress, dtype="bool"),
                                  tvm.const(top_k, dtype="int32"),
                                  tvm.const(coord_start, dtype="int32"),
                                  tvm.const(id_index, dtype="int32"),
                                  tvm.const(score_index, dtype="int32"))
    if not return_indices and invalid_to_bottom:
        out = hybrid_rearrange_out(out)

    return box_indices if return_indices else out
