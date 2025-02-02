# Knowledge Distillation Experiment

## 概要
このプロジェクトは、知識蒸留（Knowledge Distillation）の概念実証実験を実装したものです。ハードウェア制約を考慮し、ローカルLLMの代わりにAPIベースのモデルを採用しています。

### 使用モデル
- Teacher model: o3-mini
- Student model: gpt-4o-mini

## セットアップ
```bash
poetry install
export OPENAI_API_KEY="your-api-key"
```

## 実行方法
```bash
poetry run python main.py  # 基本的な実験の実行
poetry run python test_inference.py  # 詳細なテストの実行
```

## 実験結果

### Teacher Model (o3-mini)の生成能力
日常的イベントから非日常的イベントまで、幅広いコンテキストで推論を生成できることを確認:

1. 日常的イベント
```
xEvent: X が公園で散歩する
xEffect: 心と体が和み元気になった
xIntent: 心機一転で気分転換を図る
```

2. SF的イベント
```
xEvent: X が宇宙船で火星に着陸する
xEffect: 着陸成功で資源発見
xIntent: 探査開拓を進める
```

3. ファンタジー
```
xEvent: X が魔法で敵を倒す
xEffect: 敵が一気に弱体化した
xIntent: 敵を完全に追い詰める意図
```

### Student Model (gpt-4o-mini)の評価能力
コンテキストに応じた厳密な評価を実施:

1. 日常的イベント（散歩）
- xEffect: True
- xIntent: True

2. SF的イベント（火星着陸）
- xEffect: False（現実的な制約を考慮）
- xIntent: True（探査という目的は妥当）

3. ファンタジー（魔法）
- xEffect: True（文脈内での因果関係は妥当）
- xIntent: False（現実世界の制約を考慮）

4. スポーツ（世界記録更新）
- xEffect: True
- xIntent: True

5. 超現実（時間操作）
- xEffect: True
- xIntent: False

## 評価指標
1. 生成率：各関係タイプの生成成功率
2. 妥当性：Student modelによる推論の評価率
3. 多様性：生成された推論の重複度

## 結論
- Teacher modelは幅広いコンテキストで一貫した推論を生成
- Student modelは現実世界の制約に基づいて厳密に評価
- 特に非日常的なイベントに対して、より厳格な評価基準を適用

