## training tensorflow2

## mnist training outputs
https://flas.mybluemix.net/

## data image.
```
date  dataset optimizer   decay  dropout modelname  epochs  testloss  testaccuracy  trainloss  trainaccuracy
0  2020-02-19 00:23:32.093058+00:00  fashion      adam  0.0001      0.1       cnn       5  0.339908        0.8764   0.415580       0.851933
1  2020-02-19 00:23:32.093058+00:00  fashion      adam  0.0001      0.1       cnn       5  0.285015        0.8964   0.284345       0.897317
2  2020-02-19 00:23:32.093058+00:00  fashion      adam  0.0001      0.1       cnn       5  0.258369        0.9041   0.241386       0.910750
3  2020-02-19 00:23:32.093058+00:00  fashion      adam  0.0001      0.1       cnn       5  0.259665        0.9070   0.209770       0.923317
4  2020-02-19 00:23:32.093058+00:00  fashion      adam  0.0001      0.1       cnn       5  0.245806        0.9118   0.185306       0.930850
5  2020-02-19 00:19:36.961977+00:00    mnist      adam  0.0001      0.1       cnn       5  0.048133        0.9849   0.150790       0.954783
6  2020-02-19 00:19:36.961977+00:00    mnist      adam  0.0001      0.1       cnn       5  0.035562        0.9886   0.050231       0.984733
7  2020-02-19 00:19:36.961977+00:00    mnist      adam  0.0001      0.1       cnn       5  0.031806        0.9892   0.033178       0.989467
8  2020-02-19 00:19:36.961977+00:00    mnist      adam  0.0001      0.1       cnn       5  0.036848        0.9877   0.022329       0.992967
9  2020-02-19 00:19:36.961977+00:00    mnist      adam  0.0001      0.1       cnn       5  0.029310        0.9900   0.016288       0.995017
10 2020-02-19 00:20:11.623381+00:00  fashion      adam  0.0001      0.1     basic       5  0.442950        0.8435   0.511553       0.821717
11 2020-02-19 00:20:11.623381+00:00  fashion      adam  0.0001      0.1     basic       5  0.384065        0.8622   0.384469       0.860250
12 2020-02-19 00:20:11.623381+00:00  fashion      adam  0.0001      0.1     basic       5  0.362425        0.8699   0.344704       0.874333
13 2020-02-19 00:20:11.623381+00:00  fashion      adam  0.0001      0.1     basic       5  0.345644        0.8790   0.317989       0.884983
14 2020-02-19 00:20:11.623381+00:00  fashion      adam  0.0001      0.1     basic       5  0.355729        0.8743   0.301662       0.889000
15 2020-02-19 00:15:52.176872+00:00    mnist      adam  0.0001      0.1     basic       5  0.143214        0.9577   0.276101       0.921500
16 2020-02-19 00:15:52.176872+00:00    mnist      adam  0.0001      0.1     basic       5  0.104236        0.9693   0.129783       0.961917
17 2020-02-19 00:15:52.176872+00:00    mnist      adam  0.0001      0.1     basic       5  0.092591        0.9712   0.094647       0.972167
18 2020-02-19 00:15:52.176872+00:00    mnist      adam  0.0001      0.1     basic       5  0.083093        0.9745   0.074229       0.977717
19 2020-02-19 00:15:52.176872+00:00    mnist      adam  0.0001      0.1     basic       5  0.077300        0.9758   0.061136       0.981267
```

## mnist
1. 1.py keras simple model.  
1. 2.py keras simple model with matplotlib.  
1. 3.py keras simple model with matplotlib on Colaboratory.  
1. 4.py keras cnn model with matplotlib on Colaboratory.  
1. 5.py keras cnn model with matplotlib.  

## keras optimizers
SGD: old and basic.  
Adagrad: L2ノルムベース  
RMSprop: L2ノルムベース、AdaGradを改良  
Adadelta: AdaGradやRMSPropを改良  
Adam: RMSProp and Momentum.  
Nadam: RMSProp and NAG(Nesterov's accelerated gradient).  
Adamax: 無限ノルムに基づくAdamの拡張  

## bigquery
1. 0.py for auth on Colaboratory.  
1. 1.py bigquery, pandas, read and write sample.  
1. 2.py mnist, adam, pandas, bigquery on Colaboratory.  

## install
```
pip install pygal
pip install ibmcloudenv
pip install pandas_gbq
```

## Colaboratory
https://colab.research.google.com/

## IBM Cloud
https://cloud.ibm.com/

## bigquery
https://cloud.google.com/bigquery
