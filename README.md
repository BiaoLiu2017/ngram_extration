# ngram_extration
## 概述  
- 目的  
用于提取语料库的最大词频情况下的最长字符串，即用于挖掘语料库中的高频长字符串，这类字符串可能是脏文本，也可能是特征文本；  
示例：  
corpus.txt中存在大量的高频字符串，如：  
```text
<sprite=3>  
【坐标分享】3级土地 (777,225)  
新技术！苹果手机三年冲的钱退回，你和代褪失败可以，不影响游戏，威信T9883K  
1级码头-(372,737) 
``` 
通过ngram_extration可以快速挖掘这类特征字符串。  

- 安装  
```shell script
pip install ngram_extration
```

- 用法  
```python
from ngram_extration.ngram_extration import NgramExtration

def load_list_from_text(path):
    res = []
    with open(path) as f:
        for l in f:
            res.append(l[:-1])
    return res

ngram_extration = NgramExtration()
contents = load_list_from_text('corpus.txt')
final_feature_list = ngram_extration.extract_ngram(contents, min_df=10, ngram_range=(5,30), analyzer='char', lowercase=False)
```
