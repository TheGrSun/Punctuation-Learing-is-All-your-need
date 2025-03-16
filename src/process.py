import os
import re
import json
import yaml
import time
import argparse
from utils import PUNCTUATION_MAP
import gc
from tqdm import tqdm
import psutil
import sys
import ijson  # 用于流式解析JSON

os.environ["export LARGE_DATA_MODE"] = "1"

TQDM_SETTINGS = {
    'unit': 'char',
    'ncols': 100,
    'ascii': True,
    'leave': True,
    'bar_format': '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
}


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def clean_text(text):
    lines = text.split('\n')
    cleaned_lines = []

    for line in lines:
        # 移除HTML标签
        line = re.sub(r'<[^>]+>', '', line)

        # 移除连续的空白字符
        line = re.sub(r'\s+', ' ', line).strip()

        if line:  # 只添加非空行
            cleaned_lines.append(line)

    return '\n'.join(cleaned_lines)


def normalize_text(text):
    # 编码为UTF-8
    text = text.encode('utf-8', 'ignore').decode('utf-8')
    # 移除所有空白字符
    text = re.sub(r'\s+', '', text)

    return text


def split_sentences(text, max_length, window_size):
    """将文本拆分为句子，使用滑动窗口和优化合并短句的方法"""
    # 确保参数是整数
    max_length = int(max_length)
    window_size = int(window_size)

    text_length = len(text)

    # 设置关键参数
    min_advance = max(int(max_length * 0.1), 10)  # 最小前进长度
    min_sentence_length = int(max_length * 0.2)  # 最小句子长度
    optimal_length = int(max_length * 0.7)  # 目标长度
    max_search_length = min(int(max_length * 1.5), text_length)  # 搜索范围上限

    # 句子切分标点符号优先级
    high_priority_puncts = ['。', '！', '？', '；']  # 高优先级：句子完整结束标点
    medium_priority_puncts = ['：', '…', ')', '》', '】', '\u201d', '\u2019', '\n']  # 中优先级：次要句末标点
    low_priority_puncts = ['，', '、', '\u201c', '\u2018', '（', '《', '【', '(']  # 低优先级：非句末标点

    # 保存句子信息
    sentence_info = []

    # 添加超时机制
    start_time = time.time()
    max_processing_time = 600  # 最多处理10分钟
    max_stall_count = 3  # 最多允许连续停滞次数

    pos = 0
    last_pos = 0  # 记录上一次处理的位置，用于检测进度
    stall_count = 0  # 记录停滞次数
    last_progress_time = time.time()  # 记录上次有进展的时间

    # 创建进度条
    current_progress_bar = tqdm(total=text_length, **TQDM_SETTINGS)
    current_progress_bar.set_description("分句")

    # 每处理一定字符数进行垃圾回收
    gc_interval = 1000000  # 每处理100万字符进行一次垃圾回收
    last_gc_pos = 0

    while pos < text_length:
        # 检查是否超时
        current_time = time.time()
        if current_time - start_time > max_processing_time:
            print(f"警告: 分句处理超时，已处理 {pos}/{text_length} 字符 ({pos / text_length * 100:.1f}%)")
            break

        # 检查是否长时间无进展
        if current_time - last_progress_time > 20:  # 无进展检测时间
            print(f"警告: 分句处理长时间无进展，前进")
            pos += min(2000, text_length - pos)  # 强制前进的字符数
            last_progress_time = current_time
            last_pos = pos
            continue

        # 检测是否停滞
        if pos == last_pos:
            stall_count += 1
            if stall_count > max_stall_count:  # 如果连续多次没有进展
                print(f"警告: 分句处理停滞 {stall_count} 次，前进")
                pos += min_advance  # 前进
                stall_count = 0
                last_progress_time = current_time
                continue
        else:
            last_pos = pos
            stall_count = 0
            last_progress_time = current_time

        # 计算本次搜索范围
        end_target = pos + optimal_length  # 理想的结束位置
        search_end = min(pos + max_search_length, text_length)  # 最大搜索范围

        # 查找最佳分割点
        best_pos = -1
        best_priority = 0  # 0=未找到, 1=低优先级, 2=中优先级, 3=高优先级

        # 优先在目标长度附近查找高优先级标点
        for i in range(min(end_target, text_length), min(end_target + window_size, text_length)):
            if i < text_length and text[i] in high_priority_puncts:
                best_pos = i + 1  # 包含标点
                best_priority = 3
                break

        # 如果没找到高优先级标点，在更大范围内寻找
        if best_priority < 3:
            # 向后扩展搜索范围，寻找高优先级标点
            for i in range(min(end_target + window_size // 2, text_length), min(search_end, text_length)):
                if i < text_length and text[i] in high_priority_puncts:
                    best_pos = i + 1
                    best_priority = 3
                    break

            # 如果仍未找到，查找中优先级标点
            if best_priority < 3:
                for i in range(min(end_target, text_length), min(search_end, text_length)):
                    if i < text_length and text[i] in medium_priority_puncts:
                        best_pos = i + 1
                        best_priority = 2
                        break

                # 最后查找低优先级标点
                if best_priority < 2:
                    for i in range(min(end_target, text_length), min(search_end, text_length)):
                        if i < text_length and text[i] in low_priority_puncts:
                            best_pos = i + 1
                            best_priority = 1
                            break

        # 如果找不到任何合适的分割点，使用最大长度强制分割
        if best_pos == -1:
            # 尝试在最大长度附近找到一个词的边界（非字母数字字符）
            for i in range(min(pos + max_length, text_length), pos, -1):
                if i < text_length and not text[i].isalnum():
                    best_pos = i + 1
                    break

            # 如果仍然找不到，使用最大长度强制分割
            if best_pos == -1:
                best_pos = min(pos + max_length, text_length)

        # 简化引号和括号配对逻辑，减少处理复杂度
        # 只检查最基本的配对，避免复杂的嵌套处理
        pairing_start_time = time.time()
        max_pairing_time = 3  # 减少配对处理时间到3秒

        # 如果分割点会导致引号或括号不配对，尝试调整
        # 不进行复杂的嵌套处理，避免处理时间过长
        if best_pos < text_length:
            # 简单检查是否有未闭合的引号
            left_quote_count = text[pos:best_pos].count('\u201c') + text[pos:best_pos].count('\u2018')
            right_quote_count = text[pos:best_pos].count('\u201d') + text[pos:best_pos].count('\u2019')

            # 如果引号不配对，尝试向前或向后调整分割点
            if left_quote_count != right_quote_count:
                # 向后最多搜索50个字符
                extended_search = min(best_pos + 50, text_length)
                for i in range(best_pos, extended_search):
                    if i >= text_length:
                        break
                    if text[i] in ['\u201d', '\u2019']:
                        best_pos = i + 1
                        break

                    # 如果搜索超时，放弃调整
                    if time.time() - pairing_start_time > max_pairing_time:
                        print("警告: 引号配对搜索超时，使用原始分割点")
                        break

        # 确保句子长度不会超过最大允许长度
        if best_pos - pos > max_length:
            # 如果超长，向前查找最近的可分割点
            for j in range(pos + max_length, pos, -1):
                if j < text_length and text[j] in high_priority_puncts + medium_priority_puncts + low_priority_puncts:
                    best_pos = j + 1
                    break
            # 如果仍然找不到，强制在最大长度处截断
            if best_pos - pos > max_length:
                best_pos = pos + max_length

        # 提取句子
        sentence = text[pos:best_pos]

        if sentence:
            # 记录句子信息
            sentence_info.append({
                "text": sentence,
                "start": pos,
                "end": best_pos,
                "length": len(sentence),
                "priority": best_priority
            })

        # 计算下一个起始位置，确保进度前进
        overlap = min(window_size // 8, 10)  # 减少重叠范围，加快处理速度
        new_pos = max(best_pos - overlap, pos + min_advance)

        # 防止无进展
        if new_pos <= pos:
            new_pos = pos + min_advance
            print(f"警告: 分句处理无进展，强制前进 {min_advance} 字符")

        pos = new_pos

        # 进度条
        current_progress_bar.update(pos - current_progress_bar.n)

        elapsed = time.time() - start_time
        if elapsed > 0:
            chars_per_sec = pos / elapsed
            current_progress_bar.set_postfix({
                '速度': f'{int(chars_per_sec)}字/秒',
                '句子': len(sentence_info)
            })

        # 定期进行垃圾回收
        if pos - last_gc_pos > gc_interval:
            gc.collect()
            last_gc_pos = pos

        if pos >= text_length:
            break

    # 合并短句子处理
    # 超时检测
    merge_start_time = time.time()
    max_merge_time = 30  # 减少合并处理时间到30秒

    print(f"分句完成，共得到 {len(sentence_info)} 个句子，开始合并短句...")

    # 先合并长度过短的句子
    merged_sentences = []
    i = 0

    # 创建合并进度条
    merge_progress_bar = tqdm(total=len(sentence_info), **TQDM_SETTINGS)
    merge_progress_bar.set_description("合并短句")

    while i < len(sentence_info):
        # 检查是否超时
        if time.time() - merge_start_time > max_merge_time:
            print(f"警告: 合并短句子超时，已处理 {i}/{len(sentence_info)} 个句子")
            # 将剩余句子直接添加到结果中
            merged_sentences.extend(sentence_info[i:])
            break

        current = sentence_info[i]

        # 跳过长度为0的句子
        if current["length"] == 0:
            i += 1
            merge_progress_bar.update(1)
            continue

        # 如果当前句子太短且不是最后一句，尝试向后合并
        if current["length"] < min_sentence_length and i < len(sentence_info) - 1:
            # 尝试合并多个短句
            combined_sentence = current.copy()
            j = i + 1

            while j < len(sentence_info) and j < i + 3:  # 最多尝试合并3个句子
                next_sentence = sentence_info[j]
                # 计算合并后的长度
                new_length = combined_sentence["length"] + next_sentence["length"]

                # 如果合并后不超过最大长度，则合并
                if new_length <= max_length:
                    combined_sentence["text"] += next_sentence["text"]
                    combined_sentence["end"] = next_sentence["end"]
                    combined_sentence["length"] = new_length
                    combined_sentence["priority"] = max(combined_sentence["priority"], next_sentence["priority"])
                    j += 1
                else:
                    break

            # 如果成功合并了句子
            if j > i + 1:
                merged_sentences.append(combined_sentence)
                merge_progress_bar.update(j - i)
                i = j
            else:
                merged_sentences.append(current)
                merge_progress_bar.update(1)
                i += 1
        else:
            merged_sentences.append(current)
            merge_progress_bar.update(1)
            i += 1

    merge_progress_bar.close()

    # 检查是否超时
    if time.time() - merge_start_time > max_merge_time:
        print("警告: 跳过优化合并阶段，直接返回结果")
        # 提取最终结果
        final_sentences = [s["text"] for s in merged_sentences]
        current_progress_bar.close()
        return final_sentences

    # 直接使用合并后的结果
    final_sentences = [s["text"] for s in merged_sentences]

    # 关闭进度条
    current_progress_bar.close()

    # 最后进行一次垃圾回收
    gc.collect()

    print(f"分句处理完成，共得到 {len(final_sentences)} 个句子")

    return final_sentences


def process_and_tag_data(text, max_seq_length, window_size):
    """处理并标记文本数据"""
    max_seq_length = int(max_seq_length)
    window_size = int(window_size)

    # 规范化文本
    normalized_text = normalize_text(text)

    # 分割句子
    sentences = split_sentences(normalized_text, max_seq_length, window_size)

    # 初始化标记数据列表
    tagged_data = []

    # 超时机制
    start_time = time.time()
    max_processing_time = 1800  # 最多处理30分钟

    # 创建进度条
    current_progress_bar = tqdm(total=len(sentences), **TQDM_SETTINGS)
    current_progress_bar.set_description("标记")

    for sentence_idx, sentence in enumerate(sentences):
        # 检查总体处理时间
        if time.time() - start_time > max_processing_time:
            print(f"警告: 处理超时，已处理 {sentence_idx}/{len(sentences)} 个句子")
            break

        if not sentence.strip():  # 忽略空句
            current_progress_bar.update(1)
            continue

        # 无标点且无空格文本
        text_no_punc = ''.join(
            c for c in sentence if c not in PUNCTUATION_MAP and not c.isspace()
        )

        # 按字符位置标注标签
        gap_labels = [[] for _ in range(len(text_no_punc))]

        # 初始化所有间隙为"O"标签
        for i in range(len(gap_labels)):
            gap_labels[i] = ["O"]

        # 处理标点逻辑
        text_index = 0  # 跟踪无标点文本的索引
        i = 0  # 跟踪原始文本的索引

        # 使用栈匹配成对标点
        left_quotes = []  # 左引号栈
        left_brackets = []  # 左括号栈

        # 添加处理超时检测
        punct_start_time = time.time()
        max_punct_time = 15  # 标点处理最多用一半时间

        try:
            while i < len(sentence):
                # 检查是否超时
                if (i % 100 == 0) and (time.time() - punct_start_time > max_punct_time):
                    print(f"警告: 句子 {sentence_idx + 1} 标点处理耗时过长，简化处理")
                    # 跳过剩余标点处理
                    break

                char = sentence[i]

                if char in PUNCTUATION_MAP:
                    tag = PUNCTUATION_MAP[char]

                    # 确定标点应该放在哪个间隙
                    if i > 0 and not sentence[i - 1].isspace() and not sentence[i - 1] in PUNCTUATION_MAP:
                        # 标点前面是文字，放在当前字符对应的位置
                        target_gap = text_index - 1  # 不计算句首位置
                    else:
                        # 标点前面是空格或其他标点，或是句首
                        if text_index > 0:  # 不是句首
                            target_gap = text_index - 1
                        else:
                            # 句首标点，跳过处理
                            i += 1
                            continue

                    # 防止索引越界
                    if 0 <= target_gap < len(gap_labels):
                        # 添加新标签，替换"O"
                        if gap_labels[target_gap] == ["O"]:
                            gap_labels[target_gap] = [tag]
                        else:
                            # 添加新标签但避免重复
                            if tag not in gap_labels[target_gap]:
                                # 处理相邻标点的标签顺序
                                # 引号等结束符需要放在句末标点之前
                                if tag.startswith('E-') and any(t.startswith('S-') for t in gap_labels[target_gap]):
                                    # 提取不同类型的标签
                                    e_tags = [t for t in gap_labels[target_gap] if t.startswith('E-')]
                                    s_tags = [t for t in gap_labels[target_gap] if t.startswith('S-')]
                                    other_tags = [t for t in gap_labels[target_gap] if
                                                  not t.startswith('E-') and not t.startswith('S-')]

                                    # 将新的E-标签加入
                                    e_tags.append(tag)
                                    # 按照正确顺序重组标签
                                    gap_labels[target_gap] = e_tags + s_tags + other_tags
                                else:
                                    gap_labels[target_gap].append(tag)

                    # 处理成对符号的匹配
                    if tag.startswith("B-"):  # 左符号
                        if "DQUOTE" in tag:
                            left_quotes.append((tag, target_gap))
                        elif any(bracket in tag for bracket in ["BRACKET", "BOOK", "BRACE", "MBRACKET"]):
                            left_brackets.append((tag, target_gap))

                    elif tag.startswith("E-"):  # 右符号
                        # 匹配右引号
                        if "DQUOTE" in tag and left_quotes:
                            for j in range(len(left_quotes) - 1, -1, -1):
                                left_tag, left_pos = left_quotes[j]
                                if "DQUOTE" in left_tag:  # 匹配成功
                                    left_quotes.pop(j)

                        # 匹配右括号类
                        elif any(bracket in tag for bracket in
                                 ["BRACKET", "BOOK", "BRACE", "MBRACKET"]) and left_brackets:
                            bracket_type = None
                            for bracket in ["BRACKET", "BOOK", "BRACE", "MBRACKET"]:
                                if bracket in tag:
                                    bracket_type = bracket
                                    break

                            if bracket_type:
                                for j in range(len(left_brackets) - 1, -1, -1):
                                    left_tag, left_pos = left_brackets[j]
                                    if bracket_type in left_tag:  # 匹配相同类型的括号
                                        left_brackets.pop(j)
                                        break

                elif not char.isspace():
                    # 遇到非空格非标点字符，text_index向后移动
                    text_index += 1

                i += 1
        except Exception as e:
            print(f"警告: 处理句子 {sentence_idx + 1} 时出错: {str(e)}")
            # 继续处理下一个句子

        # 确保所有位置至少有一个标签
        for i in range(len(gap_labels)):
            if not gap_labels[i]:
                gap_labels[i] = ["O"]

        tagged = {
            "text": text_no_punc,
            "labels": gap_labels,
            "language": "zh"
        }

        if tagged["text"]:  # 忽略空文本
            tagged_data.append(tagged)

        # 检查总处理时间
        if time.time() - start_time > max_processing_time:
            print(f"警告: 总处理时间超过 {max_processing_time} 秒，提前结束处理")
            break

        # 进度条
        current_progress_bar.update(1)

        # 每处理100个句子更新一次进度信息
        if sentence_idx % 100 == 0:
            elapsed = time.time() - start_time
            if elapsed > 0:
                sentences_per_sec = (sentence_idx + 1) / elapsed
                current_progress_bar.set_postfix({
                    '速度': f'{sentences_per_sec:.1f}句/秒',
                    '标记数': len(tagged_data)
                })

    current_progress_bar.close()
    return tagged_data


def process_and_tag_data_in_chunks(text, max_seq_length, window_size, output_dir=None, chunk_size=None, 
                                   max_chunk_time=180, global_timeout=7200, batch_size=50, temp_dir_name="temp_chunks"):
    """将文本分块处理
    
    Args:
        text: 要处理的文本
        max_seq_length: 最大序列长度
        window_size: 滑动窗口大小
        output_dir: 输出目录
        chunk_size: 文本分块大小
        max_chunk_time: 单个块的最大处理时间(秒)
        global_timeout: 全局超时时间(秒)
        batch_size: 每处理多少个块保存一次中间结果
        temp_dir_name: 临时目录名称
    
    Returns:
        processed_chunk_files: 已处理的块文件列表
        total_items: 总数据条目数
        temp_dir: 临时目录路径
    """
    # 记录开始时间
    start_time = time.time()

    # 使用默认值
    chunk_size = chunk_size or 100000  # 默认块大小
    max_chunk_processing_time = max_chunk_time  # 单个块的最大处理时间

    # 分块策略
    text_length = len(text)
    print(f"文本总长度: {text_length} 字符，将分为多个块进行处理")
    print(f"使用块大小: {chunk_size}")

    # 创建文本块索引
    chunk_positions = []
    i = 0

    while i < text_length:
        # 计算当前块的结束位置
        end = min(i + chunk_size, text_length)

        if end < text_length:
            search_end = min(end + 200, text_length)
            for j in range(end, search_end):
                if j >= text_length or text[j] in '。！？.!?':
                    end = j + 1
                    break
        chunk_positions.append((i, end))
        i = end

    total_chunks = len(chunk_positions)
    # 计算平均块大小
    avg_chunk_size = int(sum(end - start for start, end in chunk_positions) / max(1, total_chunks))
    print(f"文本已分割为 {total_chunks} 个块 (平均每块 {avg_chunk_size} 字符)")
    gc.collect()

    # 创建进度条
    current_progress_bar = tqdm(total=text_length, **TQDM_SETTINGS)
    current_progress_bar.set_description("分块处理")

    # 流式处理所有块
    processed_chars = 0
    total_items = 0

    # 设置详细信息打印间隔
    info_print_interval = 50  # 每处理50个块打印一次详细信息

    # 创建临时文件夹存储中间结果
    temp_dir = os.path.join(output_dir or ".", temp_dir_name)
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    # 记录已处理的块文件
    processed_chunk_files = []

    # 内存监控阈值
    memory_threshold = 0.7 * psutil.virtual_memory().total

    # 流式处理每个块
    for i, (start_pos, end_pos) in enumerate(chunk_positions):
        # 检查全局超时
        if time.time() - start_time > global_timeout:
            print(f"警告: 全局处理时间超过 {global_timeout / 3600:.1f} 小时，提前结束处理")
            break

        # 检查内存使用情况
        current_memory = psutil.Process().memory_info().rss
        if current_memory > memory_threshold:
            print(f"警告: 内存使用接近阈值 ({current_memory / (1024 ** 3):.2f} GB)，强制垃圾回收")
            gc.collect()
            # 如果垃圾回收后仍然超过阈值，暂停
            if psutil.Process().memory_info().rss > memory_threshold:
                print("内存使用仍然较高，暂停5秒...")
                time.sleep(5)

        chunk_start_time = time.time()

        # 只加载当前需要处理的块
        current_chunk = text[start_pos:end_pos]

        # 处理当前块
        try:
            remaining_time = max_chunk_processing_time - (time.time() - chunk_start_time)
            if remaining_time <= 0:
                print(f"警告: 块 {i + 1}/{total_chunks} 处理超时，跳过")
                result = []
            else:
                result = process_and_tag_data(current_chunk, int(max_seq_length), int(window_size))
        except Exception as e:
            print(f"错误: 处理块 {i + 1}/{total_chunks} 时发生异常: {str(e)}")
            import traceback
            traceback.print_exc()  # 打印详细的错误堆栈
            result = []

        if time.time() - chunk_start_time > max_chunk_processing_time * 1.5:
            print(f"警告: 块 {i + 1}/{total_chunks} 处理严重超时")

        # 将结果保存到临时文件
        if isinstance(result, list) and result:
            # 创建临时文件名
            temp_file = os.path.join(temp_dir, f"chunk_{i:05d}.json")
            # 保存当前块的结果
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False)
            processed_chunk_files.append(temp_file)
            total_items += len(result)

        chunk_length = end_pos - start_pos
        processed_chars += chunk_length

        # 更新进度条和后缀信息
        current_progress_bar.update(chunk_length)
        elapsed = time.time() - start_time
        if elapsed > 0 and processed_chars > 0:
            chars_per_sec = processed_chars / elapsed
            if chars_per_sec > 0:
                remaining_time_estimate = int((text_length - processed_chars) / chars_per_sec)
            else:
                remaining_time_estimate = 0
            remaining_time_str = time.strftime("%H:%M:%S", time.gmtime(remaining_time_estimate))
            current_progress_bar.set_postfix({
                '速度': f'{int(chars_per_sec)}字符/秒',
                '已处理数据': f'已处理 {i + 1}/{total_chunks} 个块，当前数据量: {total_items} 条',
                '内存使用': f'{current_memory / (1024 * 1024):.1f} MB'
            })

        if (i + 1) % batch_size == 0:
            gc.collect()

        if (i + 1) % info_print_interval == 0:
            # 打印信息
            print(
                f"进度: {i + 1}/{total_chunks} 块 | 数据: {total_items} 条 | "
                f"内存: {psutil.Process().memory_info().rss / (1024 * 1024):.1f} MB")

    current_progress_bar.close()

    # 在所有处理完成后打印一次总结信息
    print(f"处理完成: 共 {len(processed_chunk_files)} 个块文件，包含 {total_items} 条数据")

    # 返回临时文件列表
    return processed_chunk_files, total_items, temp_dir


def read_data(file_path):
    """读取文本文件"""
    global current_progress_bar, progress_position
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到文件: {file_path}")

    # 获取文件大小
    file_size = os.path.getsize(file_path) / (1024 * 1024)  # 转换为MB
    print(f"读取文件: {file_path} (大小: {file_size:.2f} MB)")

    # 使用内存映射读取大文件
    if file_size > 100:
        print("文件较大，使用内存映射读取...")

        # 创建进度条
        current_progress_bar = tqdm(total=os.path.getsize(file_path), **TQDM_SETTINGS)
        current_progress_bar.set_description(f"读取文件: {os.path.basename(file_path)}")

        # 使用生成器模式读取文件
        def read_in_chunks(file_object, chunk_size=1024 * 1024):
            """生成器函数，分块读取文件"""
            while True:
                data = file_object.read(chunk_size)
                if not data:
                    break
                yield data

        # 分块读取并处理文件
        text_chunks = []
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            for chunk in read_in_chunks(f):
                text_chunks.append(chunk)
                # 更新进度条
                current_progress_bar.update(len(chunk.encode('utf-8')))

                # 如果累积的文本块太多，合并它们以释放内存
                if len(text_chunks) > 10:
                    text_chunks = [''.join(text_chunks)]
                    gc.collect()

        # 合并所有文本块
        text = ''.join(text_chunks)
        current_progress_bar.close()
    else:
        # 对于小文件，使用常规方法读取
        text = []
        with open(file_path, 'r', encoding='utf-8') as f:
            # 计算总行数
            total_lines = sum(1 for _ in f)
            f.seek(0)

            # 创建进度条
            current_progress_bar = tqdm(total=total_lines, **TQDM_SETTINGS)
            current_progress_bar.set_description(f"读取文件: {os.path.basename(file_path)}")

            # 读取文件
            for line in f:
                text.append(line.strip())
                # 更新进度条
                current_progress_bar.update(1)

            # 关闭进度条
            current_progress_bar.close()

        # 合并文本
        text = '\n'.join(text)

    # 垃圾回收
    gc.collect()

    return text


def save_to_json_batch(data, filename, output_dir, batch_size=5000, buffer_size=16*1024*1024):
    """批量保存数据到JSON文件
    
    Args:
        data: 要保存的数据
        filename: 文件名
        output_dir: 输出目录
        batch_size: 每批处理的记录数
        buffer_size: 文件写入缓冲区大小
    """
    filepath = os.path.join(output_dir, filename)
    total_items = len(data)

    # 创建进度条
    progress_bar = tqdm(total=total_items, **TQDM_SETTINGS)
    progress_bar.set_description(f"保存 {filename}")

    with open(filepath, 'w', encoding='utf-8', buffering=buffer_size) as f:
        f.write('[\n')
        for i in range(0, total_items, batch_size):
            batch_end = min(i + batch_size, total_items)
            batch = data[i:batch_end]

            for j, item in enumerate(batch):
                if i > 0 or j > 0:
                    f.write(',\n')
                json.dump(item, f, ensure_ascii=False)  # 移除indent=2以减少文件大小
                # 更新进度条
                progress_bar.update(1)
                
                # 每100个项目更新一次进度信息
                if (j % 100 == 0):
                    progress_bar.set_postfix({'内存': memory_usage_str()})

            # 每批写入后强制刷新文件缓冲区
            f.flush()

            # 如果不是最后一批，释放当前批次内存
            if batch_end < total_items:
                # 定期进行垃圾回收
                if i % (batch_size * 5) == 0:
                    gc.collect()

        f.write('\n]')

    # 关闭进度条
    progress_bar.close()
    file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
    print(f"✓ 已保存: {filepath} ({total_items} 条记录, {file_size:.2f} MB)")


def stream_json_array(file_path, buffer_size=16*1024*1024):
    """流式读取JSON数组，逐个返回数组元素"""
    try:
        with open(file_path, 'r', encoding='utf-8', buffering=buffer_size) as f:
            # 检查文件是否是JSON数组
            first_char = f.read(1).strip()
            f.seek(0)
            
            if first_char == '[':
                # 使用ijson流式解析JSON数组
                items = ijson.items(f, 'item')
                for item in items:
                    yield item
            else:
                # 按行处理JSON对象
                for line in f:
                    line = line.strip()
                    if not line or line == '[' or line == ']':
                        continue
                    
                    # 处理可能的逗号结尾
                    if line.endswith(','):
                        line = line[:-1]
                    
                    try:
                        item = json.loads(line)
                        yield item
                    except json.JSONDecodeError:
                        print(f"警告: 跳过无效JSON行: {line[:50]}...", file=sys.stderr)
    except Exception as e:
        print(f"错误: 读取文件 {file_path} 时出错: {str(e)}", file=sys.stderr)
        yield None


def main():
    """主函数"""
    global current_progress_bar, progress_position
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="数据预处理脚本")
    parser.add_argument("--raw_data_path", type=str, default=None,
                        help="原始数据文件路径")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="输出目录路径")
    parser.add_argument("--max_processing_time", type=int, default=7200,
                        help="最大处理时间(秒)")
    parser.add_argument("--chunk_size", type=int, default=10000,
                        help="文本分块大小")
    parser.add_argument("--memory_limit", type=float, default=0.8,
                        help="内存使用限制(占总内存比例，默认0.8)")
    parser.add_argument("--keep_temp_files", action="store_true",
                        help="保留临时文件(默认删除)")
    parser.add_argument("--skip_to_merge", action="store_true",
                        help="跳过处理步骤，直接合并临时文件")
    parser.add_argument("--config", type=str, default='configs/config.yml',
                        help="配置文件路径")
    args = parser.parse_args()

    # 记录开始时间
    global_start_time = time.time()

    # 获取系统总内存
    total_memory = psutil.virtual_memory().total

    print("=" * 80)
    print("数据预处理开始")
    print(f"系统总内存: {total_memory / (1024 ** 3):.2f} GB")
    print("=" * 80)

    # 临时目录
    temp_dir = None

    try:
        # 加载配置
        print(f"正在加载配置文件: {args.config}")
        config = load_config(args.config)
        print(f"配置文件加载成功")
        max_seq_len = config['model']['max_seq_length']
        win_size = config['data']['window_size']
        
        # 从配置文件读取处理参数，如果命令行有参数则优先使用命令行参数
        process_config = config.get('process', {})
        print(f"处理配置: {process_config}")
        
        # 设置输入输出路径
        raw_data_path = args.raw_data_path
        if raw_data_path is None:
            raw_data_path = process_config.get('raw_data_path', 'data/raw/raw_data.txt')
            print(f"从配置文件读取原始数据路径: {raw_data_path}")
        else:
            print(f"使用命令行指定的数据路径: {raw_data_path}")
            
        output_dir = args.output_dir
        if output_dir is None:
            output_dir = process_config.get('output_dir', 'data/processed')
            print(f"从配置文件读取输出目录: {output_dir}")
        else:
            print(f"使用命令行指定的输出目录: {output_dir}")
            
        max_processing_time = args.max_processing_time or process_config.get('max_processing_time', 7200)
        chunk_size = args.chunk_size or process_config.get('chunk_size', 10000)
        memory_limit = args.memory_limit or process_config.get('memory_limit', 0.8)
        memory_limit = memory_limit * total_memory
        
        # 读取缓冲区大小
        buffer_size = process_config.get('buffer_size', 16*1024*1024)  # 默认16MB
        
        # 读取批处理大小
        batch_size = process_config.get('batch_size', 50)  # 默认50
        
        # 读取临时目录配置
        temp_dir_name = process_config.get('temp_dir', 'temp_chunks')
        
        # 读取数据集划分比例
        train_ratio = process_config.get('train_ratio', 0.8)
        dev_ratio = process_config.get('dev_ratio', 0.1)
        test_ratio = process_config.get('test_ratio', 0.1)
        
        # 读取文件名称配置
        train_file_prefix = process_config.get('train_file_prefix', 'train_temp_')
        dev_file_prefix = process_config.get('dev_file_prefix', 'dev_temp_')
        test_file_prefix = process_config.get('test_file_prefix', 'test_temp_')
        train_output = process_config.get('train_output', 'train.json')
        dev_output = process_config.get('dev_output', 'dev.json')
        test_output = process_config.get('test_output', 'test.json')
        
        print(f"配置信息:")
        print(f"- 原始数据路径: {raw_data_path}")
        print(f"- 输出目录: {output_dir}")
        print(f"- 最大序列长度: {max_seq_len}")
        print(f"- 滑动窗口大小: {win_size}")
        print(f"- 块大小: {chunk_size}")
        print(f"- 最大处理时间: {max_processing_time}秒")
        print(f"- 内存使用限制: {memory_limit / (1024 ** 3):.2f}GB")

        # 确保输出目录存在
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        #  读取数据
        print("\n[步骤 1/6] 读取原始文本数据")
        try:
            original_text = read_data(raw_data_path)
        except Exception as e:
            print(f"错误: 读取数据失败: {str(e)}")
            return

        original_size = len(original_text)
        original_size_mb = original_size / 1024 / 1024
        print(f"原始文本大小: {original_size_mb:.2f} MB, {original_size} 字符")

        # 检查内存使用情况
        current_memory = psutil.Process().memory_info().rss
        print(f"当前内存使用: {current_memory / (1024 ** 3):.2f} GB ({current_memory / total_memory * 100:.1f}%)")

        # 检查处理时间
        if time.time() - global_start_time > max_processing_time:
            print(f"警告: 处理时间已超过 {max_processing_time / 3600:.1f} 小时，强制结束")
            return

        # 保存原始文本的副本
        text = original_text

        # 清理文本
        print("\n[步骤 2/6] 清理和规范化文本")
        try:
            cleaned_text = clean_text(text)
            # 释放原始文本内存
            text = None
            gc.collect()
        except Exception as e:
            print(f"错误: 清理文本失败: {str(e)}")
            cleaned_text = text
            gc.collect()

        cleaned_size = len(cleaned_text)
        cleaned_size_mb = cleaned_size / 1024 / 1024
        print(f"清理后文本大小: {cleaned_size_mb:.2f} MB, {cleaned_size} 字符")
        print(f"清理后文本占原始文本的比例: {cleaned_size / original_size * 100:.2f}%")

        # 检查内存使用情况
        current_memory = psutil.Process().memory_info().rss
        print(f"当前内存使用: {current_memory / (1024 ** 3):.2f} GB ({current_memory / total_memory * 100:.1f}%)")

        # 如果内存使用超过限制，尝试释放内存
        if current_memory > memory_limit:
            print(f"警告: 内存使用超过限制，尝试释放内存")
            gc.collect()
            # 如果仍然超过限制，强制结束
            if psutil.Process().memory_info().rss > memory_limit:
                print(f"错误: 内存使用仍然超过限制，强制结束")
                return

        # 检查处理时间
        if time.time() - global_start_time > max_processing_time:
            print(f"警告: 处理时间已超过 {max_processing_time / 3600:.1f} 小时，强制结束")
            return

        # 规范化文本
        print("\n[步骤 3/6] 编码规范化")
        try:
            normalized_text = normalize_text(cleaned_text)
            gc.collect()
        except Exception as e:
            print(f"错误: 规范化文本失败: {str(e)}")
            return

        normalized_size = len(normalized_text)
        normalized_size_mb = normalized_size / 1024 / 1024
        print(f"规范化后文本大小: {normalized_size_mb:.2f} MB, {normalized_size} 字符")
        print(f"规范化后文本占原始文本的比例: {normalized_size / original_size * 100:.2f}%")

        # 检查内存使用情况
        current_memory = psutil.Process().memory_info().rss
        print(f"当前内存使用: {current_memory / (1024 ** 3):.2f} GB ({current_memory / total_memory * 100:.1f}%)")

        # 检查处理时间
        if time.time() - global_start_time > max_processing_time:
            print(f"警告: 处理时间已超过 {max_processing_time / 3600:.1f} 小时，强制结束")
            return

        # 处理数据
        print(f"\n[步骤 4/6] 处理和标记数据 (最大句长: {max_seq_len}, 滑动窗口: {win_size})")
        # 传入原始文本和处理后的文本
        try:
            # 处理文本
            processed_chunk_files, total_items, temp_dir = process_and_tag_data_in_chunks(
                normalized_text, 
                max_seq_len,
                win_size, 
                output_dir,
                chunk_size=chunk_size,
                max_chunk_time=process_config.get('max_chunk_processing_time', 180),
                global_timeout=max_processing_time,
                batch_size=process_config.get('batch_size', 50),
                temp_dir_name=temp_dir_name
            )

            gc.collect()

        except Exception as e:
            print(f"错误: 处理数据失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return

        # 检查内存使用情况
        current_memory = psutil.Process().memory_info().rss
        print(f"当前内存使用: {current_memory / (1024 ** 3):.2f} GB ({current_memory / total_memory * 100:.1f}%)")

        # 划分数据集
        print("\n[步骤 5/6] 划分训练/验证/测试集")
        
        # 如果直接跳到合并步骤
        if args.skip_to_merge:
            print("跳过处理步骤，直接合并临时文件...")
            temp_dir = os.path.join(output_dir, temp_dir_name)
            if not os.path.exists(temp_dir):
                print(f"错误: 临时目录 {temp_dir} 不存在!")
                return
                
            # 获取所有临时文件
            train_temp_files = sorted([f for f in os.listdir(temp_dir) if f.startswith(train_file_prefix)])
            dev_temp_files = sorted([f for f in os.listdir(temp_dir) if f.startswith(dev_file_prefix)])
            test_temp_files = sorted([f for f in os.listdir(temp_dir) if f.startswith(test_file_prefix)])
            
            print(f"找到 {len(train_temp_files)} 个训练集临时文件")
            print(f"找到 {len(dev_temp_files)} 个验证集临时文件")
            print(f"找到 {len(test_temp_files)} 个测试集临时文件")
            
            if not (train_temp_files or dev_temp_files or test_temp_files):
                print("错误: 没有找到任何临时文件!")
                return

            # 使用优化的方法处理临时文件
            if train_temp_files:
                process_temp_files(temp_dir, train_temp_files, os.path.join(output_dir, train_output), buffer_size=buffer_size)
                gc.collect()
            
            if dev_temp_files:
                process_temp_files(temp_dir, dev_temp_files, os.path.join(output_dir, dev_output), buffer_size=buffer_size)
                gc.collect()
            
            if test_temp_files:
                process_temp_files(temp_dir, test_temp_files, os.path.join(output_dir, test_output), buffer_size=buffer_size)
                gc.collect()
                
            # 跳到最后的统计步骤
            train_file_path = os.path.join(output_dir, train_output)
            dev_file_path = os.path.join(output_dir, dev_output)
            test_file_path = os.path.join(output_dir, test_output)

            train_file_size = os.path.getsize(train_file_path) if os.path.exists(train_file_path) else 0
            dev_file_size = os.path.getsize(dev_file_path) if os.path.exists(dev_file_path) else 0
            test_file_size = os.path.getsize(test_file_path) if os.path.exists(test_file_path) else 0

            total_processed_size = train_file_size + dev_file_size + test_file_size
            total_processed_size_mb = total_processed_size / 1024 / 1024

            print(f"处理完成! 总处理时间: {(time.time() - global_start_time) / 60:.1f} 分钟")
            print(f"输出文件总大小: {total_processed_size_mb:.2f} MB")
            print(f"输出目录: {output_dir}")

            # 最终内存使用情况
            current_memory = psutil.Process().memory_info().rss
            print(f"最终内存使用: {current_memory / (1024 ** 3):.2f} GB ({current_memory / total_memory * 100:.1f}%)")
            return
            
        train_data = []
        dev_data = []
        test_data = []

        # 创建进度条
        current_progress_bar = tqdm(total=len(processed_chunk_files), **TQDM_SETTINGS)
        current_progress_bar.set_description("划分数据集")

        # 分批处理临时文件
        file_batch_size = 5  # 每次处理5个文件
        
        for i in range(0, len(processed_chunk_files), file_batch_size):
            # 检查内存使用情况
            if i % (file_batch_size * 5) == 0 and i > 0:
                current_memory = psutil.Process().memory_info().rss
                if current_memory > memory_limit * 0.9:  # 使用90%阈值
                    print(f"警告: 内存使用接近限制 ({current_memory / (1024 ** 3):.2f} GB)，保存当前数据并释放内存")

                    # 保存当前已处理的数据
                    if len(train_data) > 0:
                        temp_train_file = os.path.join(temp_dir, f"{train_file_prefix}{i}.json")
                        with open(temp_train_file, 'w', encoding='utf-8', buffering=buffer_size) as f:
                            json.dump(train_data, f, ensure_ascii=False)
                        train_data = []

                    if len(dev_data) > 0:
                        temp_dev_file = os.path.join(temp_dir, f"{dev_file_prefix}{i}.json")
                        with open(temp_dev_file, 'w', encoding='utf-8', buffering=buffer_size) as f:
                            json.dump(dev_data, f, ensure_ascii=False)
                        dev_data = []

                    if len(test_data) > 0:
                        temp_test_file = os.path.join(temp_dir, f"{test_file_prefix}{i}.json")
                        with open(temp_test_file, 'w', encoding='utf-8', buffering=buffer_size) as f:
                            json.dump(test_data, f, ensure_ascii=False)
                        test_data = []

                    # 强制垃圾回收
                    gc.collect()

            # 获取当前批次的文件
            batch_end = min(i + file_batch_size, len(processed_chunk_files))
            batch_files = processed_chunk_files[i:batch_end]

            # 处理当前批次的文件
            for file_idx, file_path in enumerate(batch_files):
                try:
                    # 显示文件大小和内存使用情况
                    file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                    current_progress_bar.set_postfix({
                        '文件': f'{i+file_idx+1}/{len(processed_chunk_files)}',
                        '大小': f'{file_size:.1f}MB',
                        '内存': memory_usage_str()
                    })
                    
                    # 使用流式读取处理大文件
                    chunk_data = []
                    for item in stream_json_array(file_path, buffer_size):
                        if item is not None:
                            chunk_data.append(item)
                    
                    # 根据比例分配数据
                    # 计算当前文件中应该分配给各集合的数量
                    train_count = int(len(chunk_data) * train_ratio)
                    dev_count = int(len(chunk_data) * dev_ratio)

                    # 分配数据
                    train_data.extend(chunk_data[:train_count])
                    dev_data.extend(chunk_data[train_count:train_count + dev_count])
                    test_data.extend(chunk_data[train_count + dev_count:])

                except Exception as e:
                    print(f"警告: 处理文件 {file_path} 时出错: {str(e)}")

            # 更新进度条
            current_progress_bar.update(len(batch_files))

            # 每处理5批次，打印一次进度
            if i % (file_batch_size * 5) == 0 and i > 0:
                print(
                    f"已处理: {i}/{len(processed_chunk_files)} 个块，训练集: {len(train_data)}，验证集: {len(dev_data)}，测试集: {len(test_data)}")
                print(f"当前内存使用: {memory_usage_str()}")

        current_progress_bar.close()

        # 检查是否有临时分割文件
        train_temp_files = sorted([f for f in os.listdir(temp_dir) if f.startswith(train_file_prefix)])
        dev_temp_files = sorted([f for f in os.listdir(temp_dir) if f.startswith(dev_file_prefix)])
        test_temp_files = sorted([f for f in os.listdir(temp_dir) if f.startswith(test_file_prefix)])

        # 如果有临时文件，需要合并它们
        if train_temp_files or dev_temp_files or test_temp_files:
            print("发现临时文件，将合并文件...")
            
            # 处理训练集
            if train_temp_files:
                if len(train_data) > 0:
                    # 保存当前内存中的数据到临时文件
                    temp_train_file = os.path.join(temp_dir, f"{train_file_prefix}final.json")
                    with open(temp_train_file, 'w', encoding='utf-8', buffering=buffer_size) as f:
                        json.dump(train_data, f, ensure_ascii=False)
                    train_temp_files.append(f"{train_file_prefix}final.json")
                    gc.collect()
                
                process_temp_files(temp_dir, train_temp_files, os.path.join(output_dir, train_output), buffer_size=buffer_size)
                gc.collect()
            elif len(train_data) > 0:
                save_to_json_batch(train_data, train_output, output_dir, batch_size=batch_size, buffer_size=buffer_size)
                gc.collect()
            
            # 处理验证集
            if dev_temp_files:
                if len(dev_data) > 0:
                    # 保存当前内存中的数据到临时文件
                    temp_dev_file = os.path.join(temp_dir, f"{dev_file_prefix}final.json")
                    with open(temp_dev_file, 'w', encoding='utf-8', buffering=buffer_size) as f:
                        json.dump(dev_data, f, ensure_ascii=False)
                    dev_temp_files.append(f"{dev_file_prefix}final.json")
                    gc.collect()
                
                process_temp_files(temp_dir, dev_temp_files, os.path.join(output_dir, dev_output), buffer_size=buffer_size)
                gc.collect()
            elif len(dev_data) > 0:
                save_to_json_batch(dev_data, dev_output, output_dir, batch_size=batch_size, buffer_size=buffer_size)
                gc.collect()
            
            # 处理测试集
            if test_temp_files:
                if len(test_data) > 0:
                    # 保存当前内存中的数据到临时文件
                    temp_test_file = os.path.join(temp_dir, f"{test_file_prefix}final.json")
                    with open(temp_test_file, 'w', encoding='utf-8', buffering=buffer_size) as f:
                        json.dump(test_data, f, ensure_ascii=False)
                    test_temp_files.append(f"{test_file_prefix}final.json")
                    gc.collect()
                
                process_temp_files(temp_dir, test_temp_files, os.path.join(output_dir, test_output), buffer_size=buffer_size)
                gc.collect()
            elif len(test_data) > 0:
                save_to_json_batch(test_data, test_output, output_dir, batch_size=batch_size, buffer_size=buffer_size)
                gc.collect()
        else:
            # 保存数据集
            print("\n[步骤 6/6] 保存数据集")
            print(f"数据集分布: 训练集 {len(train_data)} 条, 验证集 {len(dev_data)} 条, 测试集 {len(test_data)} 条")

            try:
                if len(train_data) > 0:
                    save_to_json_batch(train_data, train_output, output_dir, batch_size=batch_size, buffer_size=buffer_size)
                    # 释放训练集内存
                    train_data = None
                    gc.collect()

                if len(dev_data) > 0:
                    save_to_json_batch(dev_data, dev_output, output_dir, batch_size=batch_size, buffer_size=buffer_size)
                    # 释放验证集内存
                    dev_data = None
                    gc.collect()

                if len(test_data) > 0:
                    save_to_json_batch(test_data, test_output, output_dir, batch_size=batch_size, buffer_size=buffer_size)
                    # 释放测试集内存
                    test_data = None
                    gc.collect()

            except Exception as e:
                print(f"错误: 保存数据集失败: {str(e)}")
                # 使用简单的JSON保存
                try:
                    if train_data and len(train_data) > 0:
                        save_to_json_batch(train_data, train_output, output_dir, batch_size=batch_size, buffer_size=buffer_size)
                        print(f"✓ 已保存: {os.path.join(output_dir, train_output)} ({len(train_data)} 条记录)")
                        gc.collect()

                    if dev_data and len(dev_data) > 0:
                        save_to_json_batch(dev_data, dev_output, output_dir, batch_size=batch_size, buffer_size=buffer_size)
                        print(f"✓ 已保存: {os.path.join(output_dir, dev_output)} ({len(dev_data)} 条记录)")
                        gc.collect()

                    if test_data and len(test_data) > 0:
                        save_to_json_batch(test_data, test_output, output_dir, batch_size=batch_size, buffer_size=buffer_size)
                        print(f"✓ 已保存: {os.path.join(output_dir, test_output)} ({len(test_data)} 条记录)")
                        gc.collect()
                except Exception as e2:
                    print(f"错误: 简单保存也失败了: {str(e2)}")
                    return

        # 计算处理后的数据总量
        train_file_path = os.path.join(output_dir, train_output)
        dev_file_path = os.path.join(output_dir, dev_output)
        test_file_path = os.path.join(output_dir, test_output)

        train_file_size = os.path.getsize(train_file_path) if os.path.exists(train_file_path) else 0
        dev_file_size = os.path.getsize(dev_file_path) if os.path.exists(dev_file_path) else 0
        test_file_size = os.path.getsize(test_file_path) if os.path.exists(test_file_path) else 0

        total_processed_size = train_file_size + dev_file_size + test_file_size
        total_processed_size_mb = total_processed_size / 1024 / 1024

        print(f"处理完成! 总处理时间: {(time.time() - global_start_time) / 60:.1f} 分钟")
        print(f"输出文件总大小: {total_processed_size_mb:.2f} MB")
        print(f"输出目录: {output_dir}")

        # 最终内存使用情况
        current_memory = psutil.Process().memory_info().rss
        print(f"最终内存使用: {current_memory / (1024 ** 3):.2f} GB ({current_memory / total_memory * 100:.1f}%)")

    except KeyboardInterrupt:
        print("\n处理被用户中断")
    except Exception as e:
        print(f"发生未预期的错误: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # 进度条关闭
        if current_progress_bar is not None:
            current_progress_bar.close()

        # 清理临时文件
        if temp_dir and os.path.exists(temp_dir) and not args.keep_temp_files:
            print(f"清理临时文件目录: {temp_dir}")
            try:
                import shutil
                shutil.rmtree(temp_dir)
                print("临时文件已清理")
            except Exception as e:
                print(f"警告: 清理临时文件失败: {str(e)}")

        # 最后进行一次垃圾回收
        gc.collect()


def process_temp_files(temp_dir, file_list, output_file, batch_size=3, buffer_size=16*1024*1024):
    """处理数据集临时文件并合并到一个输出文件
    
    Args:
        temp_dir: 临时文件目录
        file_list: 需要处理的文件列表
        output_file: 输出文件路径
        batch_size: 批处理大小
        buffer_size: 文件读写缓冲区大小
    """
    # 创建输出文件目录
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 计算总项目数（估计值）
    total_items_estimate = 0
    for temp_file in file_list[:min(3, len(file_list))]:  # 只检查前几个文件来估计
        file_path = os.path.join(temp_dir, temp_file)
        file_size = os.path.getsize(file_path)
        # 根据文件大小估计项目数
        total_items_estimate += int(file_size / 1000)  # 假设每个项目平均1KB
    
    # 创建进度条
    progress_bar = tqdm(total=total_items_estimate, **TQDM_SETTINGS)
    progress_bar.set_description(f"处理 {os.path.basename(output_file)}")
    
    # 使用流式写入
    with open(output_file, 'w', encoding='utf-8', buffering=buffer_size) as out_f:
        out_f.write('[\n')
        
        first_item = True
        total_processed = 0
        
        # 分批处理文件
        for i in range(0, len(file_list), batch_size):
            batch_files = file_list[i:i+batch_size]
            
            for file_idx, temp_file in enumerate(batch_files):
                file_path = os.path.join(temp_dir, temp_file)
                
                try:
                    # 检查文件大小
                    file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                    progress_bar.set_postfix({
                        '文件': f'{i+file_idx+1}/{len(file_list)}',
                        '大小': f'{file_size:.1f}MB',
                        '内存': memory_usage_str()
                    })
                    
                    # 流式读取和处理大文件
                    item_count = 0
                    for item in stream_json_array(file_path, buffer_size):
                        if item is None:
                            continue
                            
                        if not first_item:
                            out_f.write(',\n')
                        else:
                            first_item = False
                        
                        json.dump(item, out_f, ensure_ascii=False)
                        item_count += 1
                        
                        # 每1000个项目更新一次进度条
                        if item_count % 1000 == 0:
                            progress_bar.update(1000)
                            progress_bar.set_postfix({
                                '文件': f'{i+file_idx+1}/{len(file_list)}',
                                '项目': item_count,
                                '内存': memory_usage_str()
                            })
                            # 定期刷新文件缓冲区
                            out_f.flush()
                    
                    # 更新最终进度
                    progress_bar.update(item_count % 1000)
                    total_processed += item_count
                    
                except Exception as e:
                    progress_bar.write(f"错误: 处理文件 {temp_file} 时出错: {str(e)}")
                
            # 每批次后进行垃圾回收
            gc.collect()
        
        out_f.write('\n]')
    
    # 更新进度条总数为实际处理的项目数
    progress_bar.total = total_processed
    progress_bar.refresh()
    progress_bar.close()
    
    # 检查输出文件
    if os.path.exists(output_file):
        file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
        print(f"✓ 已保存: {output_file} ({total_processed} 条记录, {file_size:.2f} MB)")
    else:
        print(f"警告: 输出文件 {output_file} 未创建")


def memory_usage_str():
    """返回当前内存使用情况的字符串表示"""
    mem = psutil.Process().memory_info().rss / (1024 * 1024 * 1024)  # GB
    return f'{mem:.2f}GB'


if __name__ == '__main__':
    main()
