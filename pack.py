import glob
import os
import zipfile

# 1. 除外したいファイル名/ディレクトリ名のリストを定義
EXCLUDE_PATTERNS = [
    '__pycache__',
    'src\\runs\\last.pt'
]

zip_filename = 'submit.zip'
print(f"Zipファイル名: {zip_filename}")

def should_exclude(filepath: str) -> bool:
    """
    指定されたファイルパスが除外パターンに一致するかどうかをチェックする。
    """
    # Windows/Linux/macOSに対応するため、os.sepでパスを分割してチェック
    path_components = filepath.split(os.sep)
    
    for pattern in EXCLUDE_PATTERNS:
        # パスの構成要素に除外パターンが含まれていたら除外
        if pattern in path_components:
            return True
        
        if filepath.endswith(pattern):
            return True
            
    return False

def add_all(zip_file: zipfile.ZipFile, files: list[str]):
    """
    ファイルのリストをzipファイルに追加する。除外パターンに一致するものはスキップする。
    """
    for file in files:
        # 除外対象のパスかチェック
        if should_exclude(file):
            # print(f' - 除外 (Excluded): {file}')
            continue

        # ディレクトリではない、つまりファイルである場合のみ追加
        if os.path.isfile(file):
            # zip.write(ファイルパス, zipファイル内でのパス)
            zip_file.write(file, arcname=file)
            print(f' + 追加 (Added): {file}')
        else:
            # ディレクトリは追加しない（globが返すディレクトリ名）
            pass


# globで取得する全ファイルとディレクトリのリスト
all_files = glob.glob("src/**", recursive=True)

with zipfile.ZipFile(zip_filename, 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
    add_all(zipf, all_files)

print(f"\n{zip_filename} の作成が完了しました。")