import xml.etree.ElementTree as ET
from stemming.porter2 import stem
import re

is_sample = 0
data_folder = ''
sample_txt_path = ''
trec_sample_path = 'trec.5000.xml'
sample_xml_path = ''
stopword_filepath = 'englishST.txt'

stopword_path = 'englishST.txt'


def load_stopword(stopword_path):
    stopword_list = []
    with open(stopword_path, 'r', encoding='utf-8') as f1:
        stopword_list = [str(current_word).strip() for current_word in f1]
    return stopword_list


stopword_list = load_stopword(stopword_path)

def tokenisation_text(text):
    """
    寻找文本中的特征标记
    :param text:
    :return:
    """
    del_punc = '\W'  # 此符号代表匹配包括下划线、大小写字母和阿拉伯数字。
    text_nopunc = re.sub(del_punc, ' ', text)  # 把所有下划线、大小写字母和阿拉伯数字替换为空格。
    with open('a.txt', 'w+') as f1:  # 替换后写入文件
        f1.write(text_nopunc)
    tokenisation_list = text_nopunc.split()  # 去掉了下划线、大小写字母和阿拉伯数字的剩余字符都视作特征标记。
    return tokenisation_list


def lower_word(word_list):
    """
    全部转小写
    :param word_list:
    :return:
    """
    return [current_word.lower() for current_word in word_list]


def token_lower_nostop_stem_list(all_text, stopword_list):
    """
    从完整文本中抽取所有非停词词干。
    :param all_text:
    :param stopword_list:
    :return:
    """
    token_list = tokenisation_text(all_text)  # 寻找文本中的特征字符，即去掉所有下划线、大小写字母和阿拉伯数字后的列表。
    token_lowerlist = lower_word(token_list)  # 全部转小写。
    token_lowerlist_nostop = [str(current_word) for current_word in token_lowerlist if
                              str(current_word) not in stopword_list]  # 选取所有非停词
    stem_list = [stem(current_word) for current_word in
                 token_lowerlist_nostop]  # 根据所有非停词计算词干。如果单词长度小于等于2，则返回本身，否则返回单词的原始形态。

    return stem_list

def to_standard_xml(xml_file_path):
    """
    将非标准XML文件变为XML文件。
    实质是在文件头和尾加入 XML DOM 节点。
    :param xml_file_path:
    :return:
    """
    new_file_name = 'standard_' + xml_file_path.split('/')[-1]
    new_file_path = data_folder + new_file_name

    with open(new_file_path, 'w', encoding='utf-8') as new_file:
        with open(xml_file_path, 'r') as original_file:
            new_file.write('<?xml version="1.0"?>\n')
            new_file.write('<sample>\n')
            for line in original_file:
                if (len(line.strip()) == 0):
                    continue
                new_file.write(line)
            new_file.write('</sample>')
    return new_file_path


def xml_sample(xml_file_path):
    """
    读取指定的 XML 文件，并返回它的根节点。
    :param xml_file_path:
    :return:
    """
    tree = ET.parse(trec_sample_path)
    root = tree.getroot()
    return root


def merge_text(text1, text2):
    merged_text = text1 + text2
    return merged_text


def xml_all_text(stopword_list):
    """
    提取所有词干。
    sample 和 trec 的唯一区别是，sample 只有 text 属性，而 trec 有 headline 和 text 属性，从 trec 提取需要将 headline 和 text 合并。
    :param stopword_list:
    :return:
    """
    standard_xml_path = to_standard_xml(sample_xml_path) if is_sample else to_standard_xml(trec_sample_path)
    tree = ET.parse(standard_xml_path)
    root = tree.getroot()
    all_text_list = [(child.find('DOCNO').text, token_lower_nostop_stem_list(child.find('Text').text, stopword_list))
                     for child in
                     root.findall("./DOC")] if is_sample else [(child.find('DOCNO').text, token_lower_nostop_stem_list(
        merge_text(child.find('TEXT').text, child.find('HEADLINE').text), stopword_list)) for child in
                                                               root.findall("./document/DOC")]
    return all_text_list

# for xml file
all_text_list = xml_all_text(stopword_list) if is_sample else xml_all_text(stopword_list)

def inverted_index(all_text_list):
    '''
    return: # index dic[stem]={'doc_id':position}
    '''

    # get stem
    stem_set = set()
    for current_doc_tupple in all_text_list:
        for current_stem in current_doc_tupple[1]:
            stem_set.add(current_stem)

    index_dic = {}
    for current_stem in stem_set:
        current_stem_doc_position_dic = {}
        for current_doc_tupple in all_text_list:
            current_doc_id = current_doc_tupple[0]
            current_doc_text_list = current_doc_tupple[1]

            if (current_stem not in current_doc_text_list):
                continue

            position_list = [index + 1 for index in range(len(current_doc_text_list))
                             if current_doc_text_list[index] == current_stem]
            current_stem_doc_position_dic[str(current_doc_id)] = position_list
        index_dic[current_stem] = current_stem_doc_position_dic
    return index_dic

index_dic = inverted_index(all_text_list)

def output_index(index_dic, file_path):
    sorted_index_dic_list = sorted(index_dic.items(), key=lambda x: x[0], reverse=False)
    with open(file_path, 'w', encoding='utf-8') as file_1:
        for current_tupple in sorted_index_dic_list:
            current_stem = current_tupple[0]
            current_stem_doc_position_dic = current_tupple[1]
            file_1.write(current_stem)
            file_1.write(':%d\n' % len(current_stem_doc_position_dic))  # 作业要求冒号后输出出现当前词的文档数量。
            for current_doc_id, current_doc_id_position_list in current_stem_doc_position_dic.items():
                file_1.write('\t')
                file_1.write(str(current_doc_id))
                file_1.write(': ')
                current_doc_id_position_list_new = map(lambda x: str(x), current_doc_id_position_list)
                # print(','.join(current_doc_id_position_list_new))
                file_1.write(','.join(current_doc_id_position_list_new))
                file_1.write('\n')
            file_1.write('\n')
    with open(file_path + "_", 'w', encoding='utf-8') as file_2:
        for current_tupple in sorted_index_dic_list:
            current_stem = current_tupple[0]
            current_stem_doc_position_dic = current_tupple[1]
            file_2.write(current_stem)
            file_2.write(':\n')
            for current_doc_id, current_doc_id_position_list in current_stem_doc_position_dic.items():
                file_2.write('\t')
                file_2.write(str(current_doc_id))
                file_2.write(': ')
                current_doc_id_position_list_new = map(lambda x: str(x), current_doc_id_position_list)
                # print(','.join(current_doc_id_position_list_new))
                file_2.write(','.join(current_doc_id_position_list_new))
                file_2.write('\n')
            file_2.write('\n')

    return

output_index_path = 'index.txt'
output_index(index_dic, output_index_path)
# output sample
output_path = 'sample.index_' if is_sample else 'trec.index_'

def load_inverted_index(index_path):
    result_inverted_index = {}
    with open(index_path, 'r') as f1:
        for line in f1:
            line = line.strip()
            if (line.endswith(':')):
                current_stem = line.replace(':', '')
                result_inverted_index[current_stem] = {}
                continue

            if (len(line) == 0):
                continue

            temp_split_list = line.split(': ')
            current_doc_id, str_position_list = temp_split_list[0], temp_split_list[1]
            current_position_list = [int(current_position) for current_position in str_position_list.split(',')]
            result_inverted_index[current_stem][current_doc_id] = current_position_list
    return result_inverted_index

inverted_index_path = 'index.txt_'
loaded_inverted_index = load_inverted_index(inverted_index_path)

def get_doc_id_set(current_inverted_dic):
    """
    :param current_inverted_dic:
    :return:
    """
    doc_id_set = set()
    for current_stem, current_doc_position_dic in current_inverted_dic.items():
        for current_doc_id in current_doc_position_dic.keys():
            doc_id_set.add(str(current_doc_id))
    return doc_id_set


def query_word(current_inverted_dic, current_word, is_not=0):
    """
    :param current_inverted_dic:
    :param current_word:
    :param is_not:
    :return:
    """
    current_word_stem = stem(current_word.strip().lower())
    if (is_not):
        for current_index_stem, current_index_stem_position in current_inverted_dic.items():
            if (current_word_stem == current_index_stem):
                doc_id_set = get_doc_id_set(current_inverted_dic)
                stem_doc_list = set(current_index_stem_position.keys())
                return list(doc_id_set.difference(stem_doc_list))
    else:
        for current_index_stem, current_index_stem_position in current_inverted_dic.items():
            if (current_word_stem == current_index_stem):
                return list(current_index_stem_position.keys())
    #     raise RuntimeError('Query not found.')
    return []

def union_list(a, b):
    """
    取并集，并按顺序返回。
    :param a:
    :param b:
    :return:
    """
    result_list = list(set(a).union(set(b)))
    result_list.sort(key=lambda i: int(i))
    return result_list

def intersection_list(a, b):
    """
    取交集，并按顺序返回。
    :param a:
    :param b:
    :return:
    """
    result_list = list(set(a).intersection(set(b)))
    result_list.sort(key=lambda i: int(i))
    return result_list


def probability_query(doc_word_pos1, doc_word_pos2, current_distance, is_phrase=0):
    result_list = []
    for current_docid_1, current_positionlist_1 in doc_word_pos1.items():
        if (current_docid_1 not in doc_word_pos2.keys()):
            continue
        current_positionlist_2 = doc_word_pos2[current_docid_1]
        i = 0
        j = 0

        while ((i <= len(current_positionlist_1) - 1) and (j <= len(current_positionlist_2) - 1)):
            if (is_phrase):
                if (int(current_positionlist_1[i]) > int(current_positionlist_2[j]) - current_distance):
                    j += 1
                    continue
                elif (int(current_positionlist_1[i]) < int(current_positionlist_2[j]) - current_distance):
                    i += 1
                    continue
                else:
                    result_list.append(current_docid_1)
                    break
            else:
                if (int(current_positionlist_1[i]) > int(current_positionlist_2[j]) + current_distance):
                    j += 1
                    continue
                elif (int(current_positionlist_1[i]) < int(current_positionlist_2[j]) - current_distance):
                    i += 1
                    continue
                else:
                    result_list.append(current_docid_1)
                    break
    return result_list

def phrase_query(current_inverted_dic, query_phrase):
    phrase_list = [stem(current_word.lower().strip()) for current_word in query_phrase.replace('"', '').split()]
    # print(phrase_list)
    result_list = probability_query(current_inverted_dic[phrase_list[0]], current_inverted_dic[phrase_list[1]], 1,
                                    is_phrase=1)
    return result_list


def boolean_query(current_inverted_dic, current_query_word):
    is_NOT = 0
    # whether contains NOT
    if (current_query_word.startswith('NOT')):
        is_NOT = 1
        current_query_word = current_query_word.replace('NOT', '').strip()

    if (current_query_word.startswith('"') and current_query_word.endswith('"')):
        result_list = phrase_query(current_inverted_dic, current_query_word)

    elif (current_query_word.startswith('#')):
        current_query_word_list = current_query_word.split('(')
        current_word_distance = int(current_query_word_list[0].replace('#', '').strip())
        current_query_stem_list = [stem(temp_word.strip()) for temp_word in
                                   current_query_word_list[1].replace(')', '').split(',')]
        result_list = probability_query(current_inverted_dic[current_query_stem_list[0]],
                                        current_inverted_dic[current_query_stem_list[1]], current_word_distance)
    else:
        result_list = query_word(current_inverted_dic, current_query_word)

    # if contains NOT, get the difference set of the result_list from the doc_id_set
    if (is_NOT):
        doc_id_set = get_doc_id_set(current_inverted_dic)

        result_list = list(doc_id_set.difference(set(result_list)))

    return result_list

from collections import defaultdict
from math import log10


def tf_idf_weight(current_inverted_dic, query_phrase, is_stop=0, stop_word_path=None):
    """
    计算所有词的 TF_IDF 权重，并按从大到小排序。
    Term Frequency(TF) = 1 + lg (某个词出现的数量 / 总词数)
    Inverse Document Frequency(IDF) = lg(总文件数 / 出现某个词的文件)
    权重 = TF * IDF
    :param current_inverted_dic:
    :param query_phrase:
    :param is_stop:
    :param stop_word_path:
    :return:
    """
    # doc_tfidf['doc_id'] = tf_idf
    doc_tfidf = defaultdict(int)

    # N
    doc_number = len(get_doc_id_set(current_inverted_dic))

    if (is_stop == 0):
        stem_word_list = [stem(current_query_word.strip().lower()) for current_query_word in query_phrase.split()]
    else:
        stopword_list = load_stopword(stop_word_path)
        stem_word_list = [stem(current_query_word.strip().lower()) for current_query_word in query_phrase.split() if
                          current_query_word.strip().lower() not in stopword_list]

    if (len(stem_word_list) == 0 or (
            len(list(set(stem_word_list).intersection(set(current_inverted_dic.keys())))) == 0)):
        raise RuntimeError('All query word not found!')

    for current_stem in stem_word_list:

        # current_stem_doc_tf_count['doc_id'] = count
        current_stem_doc_tf_count = {}

        if (current_stem not in current_inverted_dic.keys()):
            continue

        for current_doc_id, current_doc_pos in current_inverted_dic[current_stem].items():
            current_stem_doc_tf_count[current_doc_id] = len(current_doc_pos)

        current_stem_df = len(current_inverted_dic[current_stem].keys())

        for current_doc_id, current_stem_doc_tf in current_stem_doc_tf_count.items():
            doc_tfidf[current_doc_id] += (1 + log10(current_stem_doc_tf)) * log10(doc_number / current_stem_df)

    sorted_doc_tfidf = sorted(doc_tfidf.items(), key=lambda x: x[1], reverse=True)

    return sorted_doc_tfidf

def output_query(query_result_dic, is_boolean=1):
    if (is_boolean):
        with open('results.boolean.txt', 'w', encoding='utf-8') as oq_file:
            for current_query_id, current_query_result_list in query_result_dic.items():
                for current_doc_id in current_query_result_list:
                    oq_file.write('%d,%d\n' % (int(current_query_id), int(current_doc_id)))
    else:
        with open('results.ranked.txt', 'w', encoding='utf-8') as oq_file:
            for current_query_id, current_query_result_list in query_result_dic.items():
                line = 0
                for current_docid_tfidf_tuple in current_query_result_list:
                    oq_file.write('%d,%d,%.4f\n' % (
                    int(current_query_id), int(current_docid_tfidf_tuple[0]), current_docid_tfidf_tuple[1]))
                    line = line + 1
                    if line >= 150:
                        break
    return

query_file_list = ['queries.boolean.txt', 'queries.ranked.txt']

operator_list = [' AND ', ' OR ']

for query_file_path in query_file_list:

    # query_result_dic[query_id] = current_query_result_list
    query_result_dic = {}
    if (query_file_path.split('.')[1] == 'boolean'):

        with open(query_file_path, 'r') as query_file:
            for current_query_line in query_file:

                query_result_list = []
                is_operator = 0
                current_query_line = current_query_line.strip()
                if (len(current_query_line) == 0):
                    continue
                current_query_id = current_query_line.split()[0]  # get query id
                current_query = ' '.join(current_query_line.split()[1:])  # get query text

                # check whether contains any operator: ' AND ' or ' OR '
                for current_operator in operator_list:

                    if (len(re.findall(current_operator, current_query))):
                        result_list = []
                        is_operator = 1
                        #                         current_query_split_list = current_query.split(current_operator)
                        current_query_split_list = re.split(current_operator, current_query)
                        for current_query_word in current_query_split_list:
                            result_list.append(boolean_query(loaded_inverted_index, current_query_word.strip()))
                        # AND
                        if (current_operator == ' AND '):
                            query_result_list = intersection_list(result_list[0], result_list[1])
                        # OR
                        else:
                            query_result_list = union_list(result_list[0], result_list[1])
                # no operator in query
                if (not is_operator):
                    query_result_list = boolean_query(loaded_inverted_index, current_query)

                query_result_dic[str(current_query_id)] = query_result_list
                print(current_query_line)
                print(len(query_result_list))
            output_query(query_result_dic)
    elif (query_file_path.split('.')[1] == 'ranked'):
        with open(query_file_path, 'r') as query_file:
            for current_query_line in query_file:
                current_query_id = current_query_line.split()[0]  # get query id
                current_query = ' '.join(current_query_line.split()[1:])  # get query text

                # remove punc
                del_punc = r'[\W]'  # keep _
                current_query = re.sub('&amp', ' ', current_query)
                current_query = re.sub(del_punc, ' ', current_query)
                current_result = tf_idf_weight(loaded_inverted_index, current_query.strip(), is_stop=1,
                                               stop_word_path=stopword_path)
                # with a maximum of 1000 results per query
                query_result_dic[str(current_query_id)] = current_result[0:1000]
            output_query(query_result_dic, is_boolean=0)

for temp1_tuple in query_result_dic.values():
    print(temp1_tuple[0])