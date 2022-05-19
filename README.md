# transformer-stt

Before fine-tune:

Test records of model"facebook/wav2vec2-base-960h"

Use 3579 records of malaya-dataset(529 MB)  to do the test:

The average accuracy:0.8839285608192896

Total time cost(run on colab with cpu):4638.542973041534 seconds

Average time cost(run on colab with cpu): 1.29 seconds

Detailed test results see the "orig_transformer_results.csv"document



After fine-tune:

Test records of model"RuiqianLi/wav2vec2-large-xls-r-300m-singlish-colab"

Use 3579 records of malaya-dataset(529 MB)  to do the test:

The average accuracy:0.9682434609642978

Detailed test results see the "after_singlish_finetune_results.csv"document
