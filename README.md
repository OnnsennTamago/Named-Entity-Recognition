# NER
NER - Named Entity Recognition
- Rule-based: The code will be tedious when we want to deal with complex text.

LSTM + CRF


<div align=center><img src="https://manu44.magtech.com.cn/Jwk_infotech_wk3/article/2019/2096-3467/2096-3467-3-2-90/img_5.png" width = "40%"/></div>  
  
Bidirectional LSTM   

from left to right get vector ${L}$  
from right to left get vercor $R$  
then, $L+R$ get vector $C$  
$C$ is fed to CRF


- [ ] CRF
- [ ] BERT + CRF
- [ ] Dilated-CNN + CRF 
