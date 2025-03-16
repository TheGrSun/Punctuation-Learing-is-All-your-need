TAG_TO_IDX = {
    'O': 0,
    'S-Comma': 1,
    'S-Period': 2,
    'S-Question': 3,
    'S-Exclamation': 4,
    'S-Semicolon': 5,
    'S-Colon': 6,
    'S-Pause': 7,

    'B-DQUOTE_L': 8,
    'E-DQUOTE_R': 9,

    'B-SQUOTE_L': 10,
    'E-SQUOTE_R': 11,

    'B-BRACKET_L': 12,
    'E-BRACKET_R': 13,

    'B-BOOK_L': 14,
    'E-BOOK_R': 15,

    'B-BRACE_L': 16,
    'E-BRACE_R': 17,

    'B-MBRACKET_L': 18,  
    'E-MBRACKET_R': 19,  
       
    'S-ELLIP': 20,
    'S-DASH': 21,
}

IDX_TO_TAG = {idx: tag for tag, idx in TAG_TO_IDX.items()}

for tag, idx in TAG_TO_IDX.items():
    assert 0 <= idx < 22, f"Invalid tag index: {idx} for tag {tag}"

PUNCTUATION_MAP = {
    # 单字符标点
    '，': 'S-Comma',  # 逗号
    '。': 'S-Period',  # 句号
    '？': 'S-Question',  # 问号
    '！': 'S-Exclamation',  # 感叹号
    '；': 'S-Semicolon',  # 分号
    '：': 'S-Colon',  # 冒号
    '、': 'S-Pause',  # 顿号

    # 成对标点 - 左
    '\u201c': 'B-DQUOTE_L',  # 左双引号
    '\u2018': 'B-SQUOTE_L',  # 左单引号
    '（': 'B-BRACKET_L',  # 左括号
    '《': 'B-BOOK_L',  # 左书名号
    '{': 'B-BRACE_L',  # 左大括号
    '【': 'B-MBRACKET_L',  # 左中括号

    # 成对标点 - 右
    '\u201d': 'E-DQUOTE_R',  # 右双引号
    '\u2019': 'E-SQUOTE_R',  # 右单引号
    '）': 'E-BRACKET_R',  # 右括号
    '》': 'E-BOOK_R',  # 右书名号
    '}': 'E-BRACE_R',  # 右大括号
    '】': 'E-MBRACKET_R',  # 右中括号

    '—': 'S-DASH',  # 破折号 
    '…': 'S-ELLIP',  # 省略号
    '……': 'S-ELLIP'  # 省略号
}
