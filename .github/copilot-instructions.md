# Copilot Instructions for cslrtools-fluentsigners50

## 概要
このリポジトリは、PyTorch/Lightningベースの連続手話認識（CSLR）タスク向けに FluentSigners50 データセットを効率的に扱うツールキットです。主な機能はデータセットのロード、アノテーションの管理、PyTorch Dataset/Lightning DataModule の提供です。

## 主要コンポーネント
- `src/cslrtools_fluentsigners50/pytorch.py`: FluentSigners50 データセットのロード・前処理・保存ロジック。`FluentSigners50` クラスが中心。
- `src/cslrtools_fluentsigners50/lightning.py`: PyTorch Lightning 用の `FluentSigners50DataModule` 実装。クロスバリデーション分割もサポート。
- `src/cslrtools_fluentsigners50/__main__.py`: コマンドラインエントリーポイント。`clipar` で引数管理。
- `src/cslrtools_fluentsigners50/v2/`: v2 API。`FS50Item`/`Metadata` など型安全なデータ表現。

## データフロー
1. データセットルート・ランドマーク・アノテーションCSVを指定してロード
2. `FluentSigners50` クラスで PyTorch Dataset として利用可能
3. 必要に応じて pickle で保存・再ロード
4. LightningDataModule で学習/検証分割や DataLoader 設定

## ワークフロー
- **インストール**: `pip install .`（Lightning利用時は `pip install .[lightning]`）
- **コマンドライン実行例**:
  ```sh
  python -m cslrtools_fluentsigners50 <dataset_root> <landmarks_dir> <output_pickle_file>
  ```
- **Python API例**:
  ```python
  from cslrtools_fluentsigners50 import FluentSigners50
  dataset = FluentSigners50(root="FluentSigners50", landmarks="landmarks")
  dataset.save("dataset.pkl")
  loaded = FluentSigners50.load("dataset.pkl")
  ```
- **テスト**: `test.py` で基本的な動作確認（pytest等は未導入）

## プロジェクト固有の慣習
- データセットのランドマークディレクトリはデータセットルート配下である必要あり（CLI引数で例外可）
- アノテーションは `gloss_annotation.csv`/`russian_translation.csv` を利用
- 並列データロード（`ThreadPoolExecutor`）で高速化
- 型安全なデータ表現（TypedDict, dataclass, TypeVar など）
- LightningDataModule の分割は人物単位でクロスバリデーション

## 依存関係
- Python >= 3.11
- torch >= 2.0.0
- numpy >= 2.3.0
- tqdm, parse, halo, cslrtools, lightning.pytorch (オプション)

## 参考ファイル
- `README.md`: 全体概要・利用例
- `src/cslrtools_fluentsigners50/pytorch.py`: データセットロジック
- `src/cslrtools_fluentsigners50/lightning.py`: Lightning連携
- `src/cslrtools_fluentsigners50/__main__.py`: CLI実装
- `src/cslrtools_fluentsigners50/v2/`: 型安全API

---
不明点や追加したい慣習があればフィードバックしてください。