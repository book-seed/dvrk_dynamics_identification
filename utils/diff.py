import numpy as np

def central_diff(array, div, order='4th_order_precision'):
    """
    计算一阶中心差分，支持二阶或四阶精度。

    参数：
    array (np.ndarray): 输入数组
    div (float): 网格间距
    order (int): 精度阶数，二阶精度或四阶精度
    """
    size = len(array)
    diff = np.zeros_like(array)
    if order == '2nd_order_precision':
        # 二阶精度：端点使用单边差分，中间点使用三点中心差分
        diff[0] = (array[1] - array[0]) / div  # 向前差分
        for i in range(1, size - 1):
            diff[i] = (array[i + 1] - array[i - 1]) / (2 * div)  # 三点中心差分
        diff[size - 1] = (array[size - 1] - array[size - 2]) / div  # 向后差分
    elif order == '4th_order_precision':
        # 四阶精度：端点使用二阶精度，内部点使用五点中心差分
        # 端点：i=0 和 i=size-1 仍使用一阶差分（与 order=1 相同）
        diff[0] = (array[1] - array[0]) / div
        diff[1] = (array[2] - array[0]) / (2 * div)  # 三点中心差分（i=1 处二阶精度）
        for i in range(2, size - 2):
            diff[i] = (-array[i + 2] + 8 * array[i + 1] - 8 * array[i - 1] + array[i - 2]) / (12 * div)  # 五点中心差分
        diff[size - 2] = (array[size - 1] - array[size - 3]) / (2 * div)  # 三点中心差分（i=size-2 处二阶精度）
        diff[size - 1] = (array[size - 1] - array[size - 2]) / div
    else:
        raise Exception('order is wrong')
    return diff


def central_2nd_diff(array, div, order='4th_order_precision'):
    """
    计算二阶中心差分，支持二阶或四阶精度。

    参数：
    array (np.ndarray): 输入数组
    div (float): 网格间距
    order (int): 精度阶数，二阶精度或四阶精度
    """
    size = len(array)
    diff = np.zeros_like(array)
    if order == '2nd_order_precision':
        # 二阶精度：仅中间点 i=1~size-2 可用三点公式，端点无法计算（保留 0 或自定义处理）
        for i in range(1, size - 1):
            diff[i] = (array[i + 1] - 2 * array[i] + array[i - 1]) / (div ** 2)
    elif order == '4th_order_precision':
        # 四阶精度：内部点 i=2~size-3 用五点公式，i=1 和 size-2 用三点公式，端点保留 0
        # 中间点（三点公式，二阶精度）
        for i in [1, size - 2]:
            diff[i] = (array[i + 1] - 2 * array[i] + array[i - 1]) / (div ** 2)
        # 内部点（五点公式，四阶精度）
        for i in range(2, size - 2):
            diff[i] = (-array[i + 2] + 16 * array[i + 1] - 30 * array[i] + 16 * array[i - 1] - array[i - 2]) / (
                        12 * (div ** 2))
    else:
        raise Exception('order is wrong')
    return diff
