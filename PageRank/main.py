from utils import *
from pagerank import *


def main():
    print('输入路径(默认为Data.txt, 缺省按回车键):')
    path = input()
    if path == '':
        path = 'Data.txt'

    print('读取数据...')
    try:
        graph, node_num, node_set = read_data(path)
    except:
        print('读取数据失败')
        return
    print('-----------读取数据成功-----------')

    if check_continuous(node_set, node_num) == False:
        return
    print('输入alpha值(默认为0.85, 缺省按回车键):')
    alpha = input()
    if alpha == '':
        alpha = 0.85
    try:
        alpha = float(alpha)
    except:
        print('输入错误')
        return

    print('选择算法(1: Basic, 2: Sparse, 3: Block, 4: BlockStripe, 5: BlockStripeParallel):')
    method = input()
    if method == '1':
        pr = PageRankBasic(graph, node_num, log=True)
    elif method == '2':
        pr = PageRankSparse(graph, node_num, log=True)
    elif method == '3':
        print('输入块大小(默认为1000, 缺省按回车键):')
        block_size = input()
        if block_size == '':
            block_size = 1000
        else:
            block_size = int(block_size)
        pr = PageRankBlock(graph, node_num, block_size, log=True)
    elif method == '4':
        print('输入块大小(默认为1000, 缺省按回车键):')
        block_size = input()
        if block_size == '':
            block_size = 1000
        else:
            block_size = int(block_size)
        pr = PRBlockStripe(graph, node_num, block_size, log=True)
    elif method == '5':
        print('输入块大小(默认为1000, 缺省按回车键):')
        block_size = input()
        if block_size == '':
            block_size = 1000
        else:
            block_size = int(block_size)
        pr = PRBlockStripeParallel(graph, node_num, block_size, log=True)
    else:
        print('输入错误')
        return
    
    scores, iteration_num = pr.power_iteration()
    sorted_indices, sorted_scores = sort_scores(scores, log=True)
    standard_result = standard_answer(graph, alpha)
    check_result(sorted_indices, sorted_scores, standard_result)
    write_result(sorted_indices, sorted_scores, 'result.txt')
    print('-----------结果已保存-----------')

if __name__ == '__main__':
    main()