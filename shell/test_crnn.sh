#!/usr/bin/env bash
export PYTHONPATH=./
#!/usr/bin/env bash
export PYTHONPATH=./
python3 './train_model/tools/test_crnn.py' \
-i='output/images/train/' \
-t='output/text_data/train_labels.txt' \
-c='output/text_data/char_map.json' \
-w='output/model_save' \
-n=10
