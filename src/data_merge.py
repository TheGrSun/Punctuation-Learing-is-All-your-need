import os
from tqdm import tqdm


def merge_txt_files(input_dir, output_file, add_filename=False, encoding='utf-8'):
    total_files = 0
    for root, _, files in os.walk(input_dir):
        total_files += len([f for f in files if f.endswith('.txt')])

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w', encoding=encoding) as outfile:
        pbar = tqdm(total=total_files, desc="合并进度", unit="file")

        for root, _, files in os.walk(input_dir):
            for filename in sorted(files):
                if filename.endswith('.txt'):
                    filepath = os.path.join(root, filename)
                    try:
                        with open(filepath, 'r', encoding=encoding) as infile:
                            content = infile.read()

                            if add_filename:
                                outfile.write(f'\n\n### {filename} ###\n\n')

                            outfile.write(content)
                            pbar.update(1)  # 更新进度条

                    except UnicodeDecodeError:
                        pbar.write(f"⚠️ 编码错误跳过: {filepath}")
                    except Exception as e:
                        pbar.write(f"⚠️ 错误处理文件: {filepath} ({str(e)})")
                    finally:
                        pbar.refresh()

        pbar.close()


if __name__ == "__main__":
    input_directory = "E:/THUCNews/THUCNews"
    output_path = 'E:/THUCNews/merged_output.txt'

    merge_txt_files(
        input_dir=input_directory,
        output_file=output_path,
        add_filename=True,
        encoding='utf-8'
    )
    print("\n✅ 合并完成！")
