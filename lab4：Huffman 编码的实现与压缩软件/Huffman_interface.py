from tkinter import *
from tkinter import filedialog
import pickle   # 可对Python的原生对象进行“打包存储”，此处用来存储全局的huffman_map
import struct   # 用来存取二进制数据
import copy


class Node(object):  # 定义结点类
    def __init__(self, symbol='', weight=0):
        self.left = None  # 左子数
        self.right = None  # 右子数
        self.symbol = symbol  # 字符
        self.weight = weight  # 权值

    def isLeaf(self):  # 判断是否为叶子节点
        return not (self.left or self.right)


# 一些全局变量
huffman_map = {}    # 存储字符与对应的编码
huffman_tree_root = Node()   # Huffman树的根节点
letter_frequency = {}     # 存储字符及其出现次数（权重）


def read_file(path):
    """
    读取文件，并统计文件中每个字符出现的次数存入全局的letter_frequency。
    :param path: 待读取文件
    :type path: str
    :return: 读取的文件内容
    :rtype: str
    """
    with open(path, 'r', encoding='utf-8') as f:
        text = ""
        for line in f.readlines():
            text += line
            for letter in line:
                if letter_frequency.get(letter):
                    letter_frequency[letter] += 1
                else:
                    letter_frequency[letter] = 1

    return text


def get_huffman_map(tree, code=''):
    """
    递归调用自身，获取每个字母的对应的Huffman编码，存入全局的huffman_map。
    :param tree: Huffman树
    :type tree: Node
    :param code: Huffman编码，一串‘0’‘1’序列
    :type code: str
    :return: None
    """
    if tree.isLeaf():
        huffman_map[tree.symbol] = code
        return
    if tree.left:
        get_huffman_map(tree.left, code + '0')
    if tree.right:
        get_huffman_map(tree.right, code + '1')


def huffman_tree():  # 构建哈夫曼树，
    """
    构建Huffman树，以嵌套的方式实现，并存入全局的huffman_tree_root中。
    :return: None
    """
    SIZE = len(letter_frequency)

    # 先将所有节点存入临时列表
    nodes = []
    for (letter, fre) in letter_frequency.items():
        nodes.append(Node(letter, fre))

    for _ in range(SIZE - 1):
        nodes.sort(key=(lambda n: n.weight))  # 以权值对节点进行排序
        left = nodes.pop(0)  # 提取权值最小的两个节点
        right = nodes.pop(0)
        parent = Node('', left.weight + right.weight)  # 创建一个新的节点
        parent.left = left  # 将之前提取的两个权值最小的节点嵌套至新结点中
        parent.right = right
        nodes.append(parent)  # 将新结点加入nodes

    if nodes:
        print(nodes[0])
        huffman_tree_root.left = nodes[0].left
        huffman_tree_root.right = nodes[0].right
        huffman_tree_root.symbol = nodes[0].symbol
        huffman_tree_root.weight = nodes[0].weight


def encode(ori_data, new_file):
    """
    对ori_data内容按照全局的letter_frequency信息进行编码，并写入new_file中
    :param ori_data: 读取到的源文件的内容
    :type ori_data: str
    :param new_file: 新文件路径名称
    :type new_file: str
    :return: None
    """

    # 编码，将形成的Huffman编码先拼成一个字符串
    coded_data = ""
    for letter in ori_data:
        coded_data += huffman_map.get(letter)

    # 将全局的letter_frequency和编码后的coded_data转为二进制，并写入new_file。
    huffman_map_bytes = pickle.dumps(huffman_map)
    f = open(new_file, 'wb')
    # 存储全局的huffman_map
    f.write(struct.pack('I%ds' % (len(huffman_map_bytes),),
                        len(huffman_map_bytes), huffman_map_bytes))

    # 存储编码后数据coded_data
    # 8位（1字节）一组进行存储，先看看是不是长度是不是8的整数倍
    f.write(struct.pack('B', len(coded_data) % 8))  # 存储最后不足一字节的二进制位数，让解码器知道有没有补0
    for index in range(0, len(coded_data), 8):
        if index + 8 < len(coded_data):
            # 将8位二进制字符串(占8字节)转换为1位十进制（占1字节）整形
            f.write(struct.pack('B', int(coded_data[index : index+8], 2)))
        else:
            # 最后若干位（<=8）转换成一个十进制整形
            f.write(struct.pack('B', int(coded_data[index:], 2)))

    f.close()
    print("Finished compressing.")


def decompress(compressed_file):
    """
    解压缩compressed_file，结果存入decompressed_file。
    :param compressed_file: 待解压文件
    :type compressed_file: str
    :param decompressed_file: 解压后存入的文件
    :type decompressed_file: str
    :return: None
    """
    f = open(compressed_file, 'rb')

    # 读取Huffman_map
    length = struct.unpack('I', f.read(4))[0]   # 前四个字节存储了huffman_map的大小
    huffman_map_byte = pickle.loads(f.read(length))     # 读出huffman_map

    # 读取二进制编码
    last_num = struct.unpack('B', f.read(1))[0]     # 读出最后不足一字节的二进制位数
    codelist = []
    data = f.read(1)
    while not data == b'':
        unpack_data = struct.unpack('B', data)
        unpack_data = unpack_data[0]
        unpack_data = bin(unpack_data)
        bdata = unpack_data[2:]
        # bdata = bin(struct.unpack('B', data)[0])[2:]
        codelist.append(bdata)
        data = f.read(1)
    f.close()

    # 将二进制数据转换为字符串
    for i in range(len(codelist) - 1):
        codelist[i] = ('0' * (8 - len(codelist[i]))) + codelist[i]
    codelist[-1] = ('0' * (last_num - len(codelist[-1]))) + codelist[-1]
    encoded_data = ''.join(codelist)    # 将datalist中的数据拼接成一个字符串

    # 将huffman_map的键值对互换
    huffman_map_byte_exchange = {}
    for (key, value) in huffman_map_byte.items():
        huffman_map_byte_exchange[value] = key

    decompressed_file = compressed_file[: -4] + "_ori.txt"
    f = open(decompressed_file, 'w', encoding='utf-8')
    letter = ''
    for b in encoded_data:
        letter += b
        if huffman_map_byte_exchange.get(letter):
            f.write(huffman_map_byte_exchange[letter])
            letter = ''
    f.close()


def compress(file):
    """
    压缩文件
    """
    # 读取待编码文件，并统计letter_frequency。
    ori_data = read_file(file)

    # 构建Huffman树
    huffman_tree()

    # 生成huffman_map，即每个字母与其Huffman编码的对应
    get_huffman_map(huffman_tree_root)

    # 输出出现频率最高的5个字符及其对应的Huffman编码，以验证编码的正确性。
    fre = sorted(letter_frequency.items(), key=lambda f: f[1], reverse=True)     # 降序排序
    for i in range(0, 3):
        print(i, fre[i][0], fre[i][1], huffman_map.get(fre[i][0]))

    # 对文章进行编码，并写入新文件
    # 新文件命名方式为：源文件 A.txt -> 新文件 A.hfm
    new_file = file[: -4] + ".hfm"
    encode(ori_data, new_file)


def Create_Interface():  # 界面
    root = Tk()
    root.title('Haffman coding')
    select_file = StringVar()  # 用于显示所选的文件
    result = StringVar()  # 用于显示执行结果

    def selectFile():
        file = filedialog.askopenfilename(filetypes=[("文本文档", '.txt'),
                                                    ('Huffmancodingfile', '.hfm')])
        select_file.set(file)

    def en():  # 编码键调用，对所选文件进行编码
        if select_file.get()[-4:] != '.txt' or not select_file.get():
            result.set('File can\'t compress.')
            return
        compress(select_file.get())
        result.set('Finished compressing.')

    def de():  # 解码键调用，对所选文件进行解码
        if select_file.get()[-4:] != '.hfm' or not select_file.get():
            result.set('File can\'t decompress.')
            return
        decompress(select_file.get())
        result.set('Finished decompressing.')

    Label(root, text="File:").grid(row=0, column=0)  # 表格排版方式
    e1 = Entry(root, textvariable=select_file, state='readonly').grid(row=0, column=1, padx=10, pady=5)  # 输入框
    Label(root, text="Information:").grid(row=1, column=0)
    e2 = Entry(root, textvariable=result, state='readonly').grid(row=1, column=1)  # 输出框
    button1 = Button(root, text="slect file", width=10, command=selectFile)  # 编码键
    button1.grid(row=0, column=2, sticky=W, padx=10, pady=5)
    button2 = Button(root, text="compress", width=10, command=en)  # 解码键
    button2.grid(row=2, column=0, sticky=W, padx=10, pady=5)
    button3 = Button(root, text="decompress", width=10, command=de)  # 解码键
    button3.grid(row=2, column=1, sticky=W, padx=10, pady=5)
    mainloop()


if __name__ == '__main__':
    Create_Interface()
